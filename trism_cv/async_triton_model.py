import os
import logging
import asyncio
import numpy as np

from tqdm.asyncio import tqdm_asyncio
from typing import Optional, Sequence, Dict, Tuple, List
from tritonclient.grpc.aio import InferInput, InferRequestedOutput

from trism_cv import client
from trism_cv.triton_types import np2trt


class AsyncTritonModel:
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

    def __init__(
        self,
        model: str,
        version: int,
        url: str,
        grpc: bool = True,
        timeout: float = 5.0,
        logger_name: Optional[str] = None,
    ) -> None:
        self._url = url
        self._grpc = grpc
        self._model = model
        self._timeout = timeout
        self._version = str(version) if version > 0 else ""

        self._serverclient = None
        self._inputs = None
        self._outputs = None

        self._initialized = False

        if logger_name:
            self.logger = logging.getLogger(f"{logger_name}.triton")
        else:
            self.logger = logging.getLogger(f"triton.{model}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the server client connection."""
        if self._serverclient is not None:
            await self._serverclient.close()
            self._serverclient = None
            self._initialized = False
            self.logger.info(f"Closed connection for model: {self._model}")

    async def _get_model_io(self) -> Tuple[List, List]:
        """
        Fetch model inputs and outputs from Triton server.
        Centralized method to avoid duplication.
        """
        return await client.iout_async(self._serverclient, self._model, self._version)

    async def setup(self) -> None:
        if self._initialized:
            return

        if self._serverclient is None:
            self._serverclient = client.serverclient_async(
                self._url, self._grpc, async_mode=True
            )

        is_ready = await self._serverclient.is_server_ready()
        if not is_ready:
            raise RuntimeError(f"Triton server at {self._url} is not ready")

        is_model_ready = await self._serverclient.is_model_ready(
            model_name=self._model, model_version=self._version
        )
        if not is_model_ready:
            raise RuntimeError(
                f"Model {self._model} (version {self._version or 'default'}) "
                f"is not ready on Triton server"
            )

        self._inputs, self._outputs = await self._get_model_io()
        self._initialized = True
        self.logger.info(f"Initialized Triton model: {self._model}")

    async def _infer_multi_inputs(
        self,
        inputs_dict: Dict[str, np.ndarray],
        timeout: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run Triton inference for multiple named inputs.

        Args:
            inputs_dict: dict of {input_name: np.ndarray}

        Returns:
            Dict[str, np.ndarray]: dict of {output_name: np.ndarray} containing model outputs.
        """

        if not self._initialized:
            await self.setup()

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

        try:
            results = await asyncio.wait_for(
                self._serverclient.infer(
                    model_name=self._model,
                    model_version=self._version,
                    inputs=infer_inputs,
                    outputs=outputs,
                ),
                timeout=timeout or self._timeout,
            )

        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Inference timeout after {timeout or self._timeout}s "
                f"for model {self._model}"
            )

        return {out.name: results.as_numpy(out.name) for out in self._outputs}

    async def run(
        self,
        data: Dict[str, Sequence[np.ndarray]],
        batch_size: int = 4,
        max_concurrent: int = 10,
        auto_config: bool = False,
        show_progress: bool = False,
    ) -> Dict[str, np.ndarray] | np.ndarray:
        """Run inference with batch processing.
        Args:
            data: Dict of model inputs, where each key is an input name and
                each value is a list of NumPy arrays (samples).
            auto_config: If True, auto-generate `config.pbtxt` from server info.
            batch_size: Number of samples per batch (auto-limited by model config).
            show_progress: If True, show progress bar using tqdm.
            max_concurrent: Number of request processing parallel

        Returns:
          - Dict[str, np.ndarray] for multi-output models.
          - np.ndarray for single-output models.
        """
        if self._inputs is None:
            await self.setup()

        if auto_config:
            await self.auto_setup_config()

        if not isinstance(data, dict):
            raise ValueError(
                "Expected a dict input, e.g. {'img': [...], 'boxes': [...]}"
            )

        # validate input key
        expected_inputs = {inp.name for inp in self._inputs}
        actual_inputs = set(data.keys())
        if not actual_inputs.issubset(expected_inputs):
            extra = actual_inputs - expected_inputs
            raise ValueError(
                f"Unexpected input keys: {extra}. Expected: {expected_inputs}"
            )
        if not actual_inputs.issuperset(expected_inputs):
            missing = expected_inputs - actual_inputs
            raise ValueError(f"Missing required input keys: {missing}")

        # process input data
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

        # Determine optimal batch size
        triton_batch_size = await self.get_max_batch_size()
        batch_size = min(triton_batch_size, batch_size)
        if batch_size < 1:
            batch_size = num_samples
        batch_indices = list(range(0, num_samples, batch_size))

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch(
            batch_idx: int, position: int
        ) -> tuple[int, Dict[str, np.ndarray]]:
            async with semaphore:
                batch_inputs = {}
                for k, v in processed_inputs.items():
                    if v.shape[0] == num_samples:
                        batch_inputs[k] = v[batch_idx : batch_idx + batch_size]
                    else:
                        batch_inputs[k] = v
                try:
                    result = await self._infer_multi_inputs(batch_inputs)
                    return position, result
                except Exception as e:
                    raise RuntimeError(
                        f"Batch at position {position} (index {batch_idx}) failed: {str(e)}"
                    ) from e

        all_outputs: list[Optional[Dict[str, np.ndarray]]] = [None] * len(batch_indices)

        tasks = [
            process_batch(batch_idx, p) for p, batch_idx in enumerate(batch_indices)
        ]
        if show_progress:
            for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
                position, res = await coro
                all_outputs[position] = res
        else:
            results = await asyncio.gather(*tasks)
            for position, res in results:
                all_outputs[position] = res

        if None in all_outputs:
            failed_positions = [i for i, o in enumerate(all_outputs) if o is None]
            raise RuntimeError(
                f"Failed to process {len(failed_positions)} batches at positions: {failed_positions}"
            )

        if len(self._outputs) > 1:
            merged = {}
            for out in self._outputs:
                all_batches = [batch[out.name] for batch in all_outputs]
                stacked = np.concatenate(all_batches, axis=0)
                merged[out.name] = stacked

            return merged
        else:
            out_name = self._outputs[0].name
            per_batch = [batch[out_name] for batch in all_outputs]
            stacked = np.concatenate(per_batch, axis=0)

            return stacked

    async def get_max_batch_size(self) -> int:
        """
        Query Triton server to get the max_batch_size for the given model.
        """
        try:
            full_config = await self._serverclient.get_model_config(
                model_name=self._model, model_version=self._version, as_json=True
            )
            config = full_config["config"]
            max_batch_size = config.get("max_batch_size", 0)

            if max_batch_size < 1:
                self.logger.warning(
                    f"Model '{self._model}' does not support batching (max_batch_size={max_batch_size})"
                )
            return max_batch_size
        except Exception as e:
            self.logger.error(
                f"Failed to get model config for '{self._model}': {e}", exc_info=True
            )
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
            self.logger.info(
                f"Config file already exists for {self._model} at {config_path}"
            )
            return

        try:
            self._inputs, self._outputs = await client.iout_async(
                self._serverclient, self.model, self._version
            )
        except Exception as e:
            self.logger.error(
                f"Failed to fetch model I/O from Triton: {e}", exc_info=True
            )
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
            self.logger.info(f"Created config.pbtxt for {self._model} at {config_path}")
        except Exception as e:
            self.logger.info(
                f"Failed to create config.pbtxt for {self._model}: {str(e)}"
            )
