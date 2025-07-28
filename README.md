# TRISM Inference Script
This script performs batch inference on a folder of images using a Triton Inference Server model.

## 📂 Input
- **image_folder**: Path to the folder containing images (jpg, png, etc.).
- Images are loaded using OpenCV (`cv2.imread`) into a list of `np.ndarray` objects.

## ⚙️ Configuration

```python
image_folder = "path/to/image_folder"   
model_name = "yolov_deyo_ensemble"
batch_size = 1 # You can pass a custom batch size or use default (e.g., 1)
```

## 🚀 Inference
```python
model = TritonModel(
    model=model_name, 
    version=1,                    # Model version on Triton server
    url="localhost:8001",         # Triton server address
    grpc=True                     # Use gRPC protocol for communication
)

outputs = model.run(
    data_list=images,              # list of images 
    auto_config=True,
    batch_size=batch_size
)
```

## 📤 Output
- A list of numpy arrays, one for each input image.
- Each output has shape `(n_detections, 6)` where `6 = [x1, y1, x2, y2, confidence, class_id]`

### 🧪 Debug Output
```python
for i, out in enumerate(outputs):
    print(f"Image {i}: shape = {out.shape}, dtype = {out.dtype}")
```

## License
[GNU AGPL v3.0](LICENSE).<br>
Copyright &copy; 2025 [Tien Nguyen Van](https://github.com/tien-ngnvan). All rights reserved.
