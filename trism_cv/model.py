import numpy as np
from typing import Optional, Dict, List
from trism_cv import client
from tritonclient.grpc import InferInput, InferRequestedOutput
from PIL import Image
import io
import cv2
import os


class Colors:
    def __init__(self):
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=True):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))

def draw_detection(image: np.ndarray, detections: List[List[float]], id2label: Optional[Dict[int, str]] = None) -> np.ndarray:
    colors = Colors()
    for *xyxy, conf, cls_id in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        cls_id = int(cls_id)
        label = id2label.get(cls_id, "Unknown") if id2label else "Unknown"
        draw_color = colors(cls_id, True)

        cv2.rectangle(image, (x1, y1), (x2, y2), draw_color, 2)
        label_text = f"{label} {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_height - 4), (x1 + text_width, y1), draw_color, -1)
        cv2.putText(image, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
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

    def run(self, image_data: List[np.ndarray], output_dir: str = "runs/predicts", save_txt: bool = False, save_image: bool = False, id2label: Optional[Dict[int, str]] = None, image_paths: Optional[List[str]] = None, max_detections: int = 100) -> Dict[str, np.ndarray]:
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
        all_detections = []
        for i, data in enumerate(image_data):

            input_data = np.expand_dims(data, axis=0)
            infer_input = InferInput("INPUT", input_data.shape, "UINT8")
            infer_input.set_data_from_numpy(input_data)
            inputs = [infer_input]
            outputs = [InferRequestedOutput("OUTPUT")]

            try:
                results = self._serverclient.infer(self.model, inputs, self._version, outputs)
                detections = results.as_numpy("OUTPUT")[0]
                num_dets = detections.shape[0]
                if num_dets > max_detections:
                    detections = detections[:max_detections]
                elif num_dets < max_detections:
                    pad = np.zeros((max_detections - num_dets, 6), dtype=np.float32)
                    detections = np.vstack([detections, pad])
                all_detections.append(detections)
            except Exception as e:
                raise

        output_dict = {"OUTPUT": np.stack(all_detections, axis=0)}

        if save_txt or save_image:
            try:
                os.makedirs(output_dir, exist_ok=True)
                if save_txt:
                    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

                for i, detections in enumerate(output_dict["OUTPUT"]):
                    file_stem = os.path.splitext(os.path.basename(image_paths[i]))[0] if image_paths else f"image_{i}"

                    if save_txt:
                        txt_path = os.path.join(output_dir, "labels", f"{file_stem}.txt")
                        try:
                            with open(txt_path, "w") as f:
                                for *xyxy, conf, cls in detections:
                                    if conf > 0:
                                        line = (*xyxy, conf, cls)
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        except Exception:
                            pass

                    if save_image:
                        try:
                            image = cv2.imread(image_paths[i])
                            if image is None:
                                pil_image = Image.open(io.BytesIO(image_data[i])).convert("RGB")
                                image = np.array(pil_image)[:, :, ::-1]
                            valid_dets = [d for d in detections if d[4] > 0]
                            image = draw_detection(image, valid_dets, id2label)
                            img_path = os.path.join(output_dir, f"{file_stem}.jpg")
                            cv2.imwrite(img_path, image)
                        except Exception:
                            pass
            except Exception:
                pass

        return output_dict
