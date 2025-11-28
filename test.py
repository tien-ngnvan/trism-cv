import os
import cv2
import numpy as np
from tqdm import tqdm
from trism_cv import TritonModel
import asyncio
# --- Cấu hình model ---
MODELS = {
    "yolov9_deyo": {
        "model_name": "detection_ensemble",
        "url": "192.168.2.21:8088",
        "classes": {0: "head", 1: "body"}
    }
}

CLASS_COLORS = {
    "body": (255, 0, 0),
    "head": (0, 255, 0)
}

OUTPUT_FOLDER = "./vis_results"
MODEL_VERSION = 1
THRESHOLD = 0.4


def load_image(img_path: str):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Failed to load image: {img_path}")
    return image


def draw_detections(image, detections, class_map):
    """Vẽ bbox lên ảnh nếu có detection hợp lệ."""
    for det in detections:
        if len(det) < 6 or det[4] < THRESHOLD:
            continue
        x1, y1, x2, y2, score, cls_id = det
        cls_id = int(cls_id)
        label = class_map.get(cls_id, str(cls_id))
        color = CLASS_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, f"{label} {score:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image


async def infer_on_triton(model, imgs, batch_size=4):


    # Padding tất cả ảnh về cùng kích thước
    max_h = max(img.shape[0] for img in imgs)
    max_w = max(img.shape[1] for img in imgs)
    padded_imgs = []
    orig_shapes = []
    for img in imgs:
        h, w = img.shape[:2]
        orig_shapes.append((h, w))
        pad_img = np.full((max_h, max_w, 3), 128, dtype=np.uint8)
        pad_img[:h, :w, :] = img
        padded_imgs.append(pad_img)

    # Chuẩn bị dict input
    data = {
        "INPUT": padded_imgs
    }

    # Gửi lên Triton
    result = await model.run(data, auto_config=False, batch_size=batch_size)


    return result


async def main_images_list(IMAGE_FOLDER):
    # Load tất cả ảnh trong folder
    img_paths = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Loaded {len(img_paths)} images")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for model_key, cfg in MODELS.items():
        print(f"\n=== Running model on images: {model_key} ===")
        triton = TritonModel(model=cfg["model_name"], version=MODEL_VERSION,
                             url=cfg["url"], grpc=True)
        class_map = cfg["classes"]
        out_dir = os.path.join(OUTPUT_FOLDER, model_key)
        os.makedirs(out_dir, exist_ok=True)

        # Load ảnh
        imgs = [load_image(p) for p in img_paths]

        # Inference trên list ảnh
        outputs = await infer_on_triton(triton, imgs, batch_size=5)
        print("output length",len(outputs))
        img_idx = 0
        for img_idx, det_output in enumerate(outputs):
            if img_idx >= len(img_paths):
                break
            img = imgs[img_idx]  
            img_with_boxes = draw_detections(img, det_output, class_map)

            out_path = os.path.join(out_dir, os.path.basename(img_paths[img_idx]))
            cv2.imwrite(out_path, img_with_boxes)
            print(f"Saved {out_path}")

if __name__ == "__main__":
    img_folder = "/home/nhattan05022003/coding/Tien_project/trism-cv-acio/human_data_sub/sub_img"
    asyncio.run(main_images_list(img_folder))
