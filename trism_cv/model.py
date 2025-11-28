import os
import cv2
import numpy as np
from tqdm.asyncio  import tqdm_asyncio
from typing import Union, Dict
from tritonclient.grpc.aio import InferInput, InferRequestedOutput

from trism_cv import client
from .types import np2trt

def load_image(img_path: str) -> np.ndarray:
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Failed to load image: {img_path}")
    return image

class TritonModel:
    @property
    def model(self) -> str:
        return self._model

    @property
    def version(self) -> str:
        return self._version

    @property
    def url(self) -> str:
        return self._url

    @property
    def grpc(self) -> bool:
        return self._grpc

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def __init__(self, model: str, version: int, url: str, grpc: bool = True) -> None:
        self._url = url
        self._grpc = grpc
        self._model = model
        self._version = str(version) if version > 0 else ""
        self._protoclient = client.protoclient(self.grpc)
        self._serverclient = None
        self._inputs = None
        self._outputs = None
    
        print("new Trism-cv version")
    
    async def setup(self):   
        if self._serverclient is None:
            self._serverclient = client.serverclient_async(
                self._url, self._grpc, async_mode=True
            )
        
        self._inputs, self._outputs = await client.iout_async(
            self._serverclient, self._model, self._version
        )

    async def _infer_multi_inputs(self, inputs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        
        """
        Run Triton inference for multiple named inputs.

        Args:
            inputs_dict: dict of {input_name: np.ndarray}
            
        Returns:
            Dict[str, np.ndarray]: dict of {output_name: np.ndarray} containing model outputs.
        """

        infer_inputs = []
        input_names = [inp.name for inp in self._inputs]
        
        for name in input_names:
            if name not in inputs_dict:
                raise KeyError(f"Missing required input '{name}' in data_dict")

            data = inputs_dict[name]
            dtype_str = np2trt(data.dtype.type)
            infer_input = InferInput(name, list(data.shape), dtype_str)
            infer_input.set_data_from_numpy(data)
            infer_inputs.append(infer_input)

        outputs = [InferRequestedOutput(out.name) for out in self._outputs]

        results = await self._serverclient.infer(
            model_name=self._model,
            model_version=self._version,
            inputs=infer_inputs,
            outputs=outputs,
        )
        return {out.name: results.as_numpy(out.name) for out in self._outputs}

    async def run(
        self, 
        data: Dict[str, Union[list[np.ndarray]]],
        batch_size: int = 2,
        auto_config: bool = False, 
        show_progress: bool = False
        ) -> Union[Dict[str, np.ndarray]]:
        """Run inference with batch processing.
        Args:
            data: Dict of model inputs, where each key is an input name and 
                each value is a list of NumPy arrays (samples).
            auto_config: If True, auto-generate `config.pbtxt` from server info.
            batch_size: Number of samples per batch (auto-limited by model config).
            show_progress: If True, show progress bar using tqdm.
        
        Returns:
          - Dict[str, np.ndarray] for multi-output models.
          - np.ndarray for single-output models.
        """
        if self._inputs is None:
            await self.setup()

        if auto_config:
            await self.auto_setup_config()

        if not isinstance(data, dict):
            raise ValueError("Expected a dict input, e.g. {'img': [...], 'boxes': [...]}")

        processed_inputs = {}
        num_samples = None

        for key, value in data.items():
            if isinstance(value, list):
                stacked = np.stack(value, axis=0)
                processed_inputs[key] = stacked
                num_samples = stacked.shape[0]
            else:
                raise TypeError(f"Unsupported type for key '{key}': {type(value)}")

        if num_samples is None:
            raise ValueError("Cannot determine batch size from input data.")

        
        triton_batch_size = await self.get_max_batch_size()
        batch_size = min(triton_batch_size, batch_size)

        iterator = range(0, num_samples, batch_size)
        if show_progress:
            iterator  = tqdm_asyncio(iterator)

        all_outputs = []
        
        for batch_idx in iterator:
            batch_inputs = {}
            for k, v in processed_inputs.items():
                if v.shape[0] == num_samples:
                    batch_inputs[k] = v[batch_idx:batch_idx + batch_size]
                else:
                    batch_inputs[k] = v  

            output = await self._infer_multi_inputs(batch_inputs)
            all_outputs.append(output)

        if len(self._outputs) > 1:
            merged = {}
            for out in self._outputs:
                all_batches = [o[out.name] for o in all_outputs]

                merged_list = []
                for batch in all_batches:
                    for i in range(batch.shape[0]):
                        merged_list.append(batch[i])

                merged[out.name] = merged_list
        else:
            out_name = self._outputs[0].name
            merged = []
            for batch in all_outputs:
                arr = batch[out_name]
                for i in range(arr.shape[0]):
                    merged.append(arr[i])
            
        return merged

    async def get_max_batch_size(self) -> int:
        """
        Query Triton server to get the max_batch_size for the given model.
        """
        try:
            full_config =await self._serverclient.get_model_config(
                model_name=self._model, model_version=self._version, as_json=True
            )
            config = full_config["config"]  
            max_batch_size = config.get("max_batch_size", 0)
            
            if max_batch_size < 1:
                print(f"Model '{self._model}' does not support batching (max_batch_size={max_batch_size})")
            return max_batch_size
        except Exception as e:
            print(f"Failed to get model config for '{self._model}': {e}")
            return 0
        

    async def auto_setup_config(self) -> None:
        """
        Automatically generate a config.pbtxt file for the Triton model if it doesn't already exist.
        Uses the real input/output info from Triton server (via self._inputs, self._outputs).
        """
        current_dir = os.getcwd()
        model_dir = os.path.join(current_dir, self._model)
        config_path = os.path.join(model_dir, "config.pbtxt")

        if os.path.exists(config_path):
            print(f"Config file already exists for {self._model} at {config_path}")
            return

        try:
            self._inputs, self._outputs = await client.iout_async(self._serverclient, self.model, self._version)
        except Exception as e:
            print(f"Failed to fetch model I/O from Triton: {e}")
            return

        input_blocks = []
        for inp in self._inputs:
            dims = getattr(inp, "shape", getattr(inp, "dims", []))
            dtype = getattr(inp, "datatype", "TYPE_FP32")
            input_blocks.append(f"""
        {{
            name: "{inp.name}"
            data_type: {dtype}
            dims: {list(dims)}
        }}""")

        output_blocks = []
        for out in self._outputs:
            dims = getattr(out, "shape", getattr(out, "dims", []))
            dtype = getattr(out, "datatype", "TYPE_FP32")
            output_blocks.append(f"""
        {{
            name: "{out.name}"
            data_type: {dtype}
            dims: {list(dims)}
        }}""")

        max_batch_size = await self.get_max_batch_size()
        config_content = f"""name: "{self._model}"
    platform: "ensemble"
    max_batch_size: {max_batch_size}

    input [{",".join(input_blocks)}
    ]

    output [{",".join(output_blocks)}
    ]
    """

        os.makedirs(model_dir, exist_ok=True)
        try:
            with open(config_path, "w") as f:
                f.write(config_content)
            print(f"Created config.pbtxt for {self._model} at {config_path}")
        except Exception as e:
            print(f"Failed to create config.pbtxt for {self._model}: {str(e)}")



