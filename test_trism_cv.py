from tqdm import tqdm
import argparse
from trism_cv.model import TritonModel
import cv2 
import os
import glob
# from trism_cv.model load_image

def load_image(img_path: str):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Failed to load image: {img_path}")
    return image

def main():
    parser = argparse.ArgumentParser(description="Run inference with TritonModel")
    parser.add_argument("--model_name", type=str, default="yolov_ensemble", help="Model name")
    parser.add_argument("--url", type=str, default="localhost:8001", help="Triton server URL")
    parser.add_argument("--data", type=str, default= "/home/nhattan05022003/coding/Tien_project/Triton_Thanh/odlab-triton/assets" ,help="Directory containing input images")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--name", type=str, default="predicts", help="Subdirectory name inside output")
    parser.add_argument("--label-file", type=str, help="Path to labels.txt file")
    parser.add_argument("--save-txt", action="store_true", help="Save detection results as text")
    parser.add_argument("--save-image", action="store_true", help="Save images with bounding boxes")
    parser.add_argument("--max-detections", type=int, default=100, help="Max detections per image")
    parser.add_argument("--auto_config", action="store_true", help="Auto-generate config.pbtxt if missing")
    parser.add_argument("--batch_size", default=3, type=int,help="Auto-generate config.pbtxt if missing")
    args = parser.parse_args()
    
    image_list = []
    image_paths = glob.glob(os.path.join(args.data, "*.*"))
    for img_path in tqdm(image_paths, desc="Loading images"):
            try:
                img = load_image(img_path)
                image_list.append(img)
            except ValueError as e:
                print(e)
                continue

    print(f"\n🚀 Inference with model: {args.model_name}")
    model = TritonModel(model=args.model_name, 
        version=1, 
        url=args.url, 
        grpc=True
        )
    
    results = model.run(
        data_list=image_list,
        auto_config = True,
        batch_size = args.batch_size
)
    print(results)

if __name__=="__main__":
     main()
