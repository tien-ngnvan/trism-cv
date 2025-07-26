import numpy as np
from typing import Optional, Dict, List
from trism_cv import client
from tritonclient.grpc import InferInput, InferRequestedOutput
import cv2
import os
from tqdm import tqdm

def load_image(img_path: str):
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
        self._serverclient = client.serverclient(self.url, self.grpc)
        self._inputs, self._outputs = client.inout(self._serverclient, self.model, self._version)

    def run(self, id2label: Optional[Dict[int, str]] = None, data_list: Optional[List[np.ndarray]] = None, auto_config=False, batch_size= int) -> Dict[str, np.ndarray]:
        """
        Run inference on a list of images and optionally save results.

        Args:
            image_data: List of image data as bytes (numpy arrays).
            output_dir: Directory to save results (images and/or text files).
            save_txt: If True, save detections as text files in YOLO format.
            save_image: If True, save images with drawn bounding boxes.
            id2label: Dictionary mapping class IDs to labels.
            image_paths: Optional list of image paths for naming output files.
            max_detections: Maximum number of detections per image (for padding).

        Returns:
            Dictionary with output tensors (e.g., {"OUTPUT": ndarray}).
        """
        print("hello")
        if auto_config:
            self.auto_setup_config()


        id2label = None
        
        if id2label:
            with open(id2label) as f:
                id2label = {i: line.strip() for i, line in enumerate(f)}

        triton_batch_size = self.get_max_batch_size()
        batch_size = min(triton_batch_size, batch_size)
        print(f"\nUsing batch_size={batch_size} (max_batch_size={triton_batch_size})")

        all_outputs = []
        for batch_idx in tqdm(range(0, len(data_list), batch_size), desc="Processing batches"):
            batch_images = data_list[batch_idx:batch_idx + batch_size]

            # Pad images to the same size within the batch
            max_height = max(img.shape[0] for img in batch_images)
            max_width = max(img.shape[1] for img in batch_images)
            padded_images = []
            for img in batch_images:
                h, w = img.shape[:2]
                padded_img = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                padded_img[:h, :w, :] = img
                padded_images.append(padded_img)

            # Create batch
            batch_data = np.stack(padded_images, axis=0)  # Shape: [batch_size, max_height, max_width, 3]

            # Send batch to Triton
            infer_input = InferInput("INPUT", batch_data.shape, "UINT8")
            infer_input.set_data_from_numpy(batch_data)

            results = self._serverclient.infer(
                model_name=self._model,
                inputs=[infer_input],
                outputs=[InferRequestedOutput("OUTPUT")],
            )
            output = results.as_numpy("OUTPUT")
            all_outputs.append(output)

        return all_outputs

        

    def get_max_batch_size(self):
        """
        Query Triton server to get the max_batch_size for the given model.
        """
        try:
            full_config = self._serverclient.get_model_config(
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


    def auto_setup_config(self, input_shape: tuple = (-1, -1, 3), output_shape: tuple = None) -> None:
        """
        Automatically generate a config.pbtxt for a Triton model if it doesn't exist.

        Args:
            model_name: Name of the model (e.g., "yolov_ensemble", "yolov_deyo_ensemble").
            input_shape: Tuple, Shape of input tensor (default: dynamic for images).
            output_shape: Tuple, Shape of output tensor (default: depends on model_name).
        """
        current_dir = os.getcwd()  
        model_dir = os.path.join(current_dir, self._model)
        config_path = os.path.join(model_dir, "config.pbtxt")
        
        if os.path.exists(config_path):
            print(f"Config file already exists for {self._model} at {config_path}")
            return
        
        # Set output shape based on model name
        if output_shape is None:
            if self._model == "yolov_ensemble":
                output_shape = (-1, 6)  # [x1, y1, x2, y2, conf, cls]
            elif self._model == "yolov_deyo_ensemble":
                output_shape = (-1, 8)  # [x1, y1, x2, y2, conf, cls, w, h]
            else:
                output_shape = (-1, 6)  # Default fallback
        
        os.makedirs(model_dir, exist_ok=True)
        
        config_content = f"""name: "{self._model}"
    platform: "ensemble"
    max_batch_size: 0
    input [
    {{
        name: "INPUT"
        data_type: TYPE_UINT8
        dims: {list(input_shape)}
    }}
    ]
    output [
    {{
        name: "OUTPUT"
        data_type: TYPE_FP32
        dims: {list(output_shape)}  
    }}
    ]
    """
        
        try:
            with open(config_path, "w") as f:
                f.write(config_content)
            print(f"Created config.pbtxt for {self._model} at {config_path}")
        except Exception as e:
            print(f"Error creating config.pbtxt for {self._model}: {str(e)}")
