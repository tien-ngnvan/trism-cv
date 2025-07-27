
# TRISM Inference Script

This script performs batch inference on a folder of images using a Triton Inference Server model.

## 📂 Input

- **image_folder**: Path to the folder containing images (jpg, png, etc.).
- Images are loaded using OpenCV (`cv2.imread`) into a list of `np.ndarray` objects.

## ⚙️ Configuration

```python
image_folder = "path/to/image_folder"   
model_name = "yolov_deyo_ensemble"
batch_size = 3 # You can pass a custom batch size or use default (e.g., 3)
```

## 🚀 Inference

### Load images:

```python
image_paths = glob.glob(os.path.join(image_folder, "*.*"))
image_list = []
for img_path in tqdm(image_paths, desc="Loading images"):
    try:
        img = load_image(img_path)  # np.ndarray (H, W, 3)
        image_list.append(img)
    except ValueError as e:
        print(f"[WARN] {e}")
        continue
```

### Initialize Triton model:

```python
model = TritonModel(
    model=model_name, 
    version=1,                     # Model version on Triton server
    url="localhost:8001",         # Triton server address
    grpc=True                     # Use gRPC protocol for communication
)
```

### Run inference:

```python
outputs = model.run(
    data_list=image_list,
    auto_config=True,
    batch_size=batch_size
)
```

## 📤 Output

- A list of numpy arrays, one for each input image.
- Each output has shape `(n_detections, 6)` where:
  - `6 = [x1, y1, x2, y2, confidence, class_id]`

## 🧪 Debug Output

```python
for i, out in enumerate(outputs):
    print(f"Image {i}: shape = {out.shape}, dtype = {out.dtype}")
```

## ✅ Requirements

- Python 3.9+
- OpenCV
- NumPy
- tqdm
- Triton Inference Server client
- `trism_cv` module (custom)


## License
[GNU AGPL v3.0](LICENSE).<br>
Copyright &copy; 2025 [Tien Nguyen Van](https://github.com/tien-ngnvan). All rights reserved.
