import numpy as np
from trism_cv.model import TritonModel
import glob
import os
import argparse



def main():
    parser = argparse.ArgumentParser(description="Run inference with TritonModel")
    parser.add_argument("--model_name", type=str, default="yolov_ensemble", help="Model name")
    parser.add_argument("--url", type=str, default="localhost:8001", help="Triton server URL")
    parser.add_argument("--data", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--name", type=str, default="predicts", help="Subdirectory name inside output")
    parser.add_argument("--label-file", type=str, help="Path to labels.txt file")
    parser.add_argument("--save-txt", action="store_true", help="Save detection results as text")
    parser.add_argument("--save-image", action="store_true", help="Save images with bounding boxes")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of images to process")
    parser.add_argument("--max-detections", type=int, default=100, help="Max detections per image")
    parser.add_argument("--auto-config", action="store_true", help="Auto-generate config.pbtxt if missing")
    args = parser.parse_args()
    
    if args.auto_config:
        TritonModel.auto_setup_config(model_name=args.model_name)


    id2label = None
    
    if args.label_file:
        with open(args.label_file) as f:
            id2label = {i: line.strip() for i, line in enumerate(f)}

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(args.data, ext)))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.data}")

    if args.batch_size is not None:
        image_paths = image_paths[:args.batch_size]

    
    image_data = []
    valid_paths = []
    for path in image_paths:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size > 0:
            image_data.append(data)
            valid_paths.append(path)

    
    output_dir = os.path.join(args.output, f"{args.name}_{args.model_name}")
    os.makedirs(output_dir, exist_ok=True)


    print(f"\n🚀 Inference with model: {args.model_name}")
    model = TritonModel(model=args.model_name, 
        version=1, 
        url=args.url, 
        grpc=True
        )
    
    # Run inference
    results = model.run(
        image_data=image_data,
        output_dir="result/predicts_yolov_ensemble",
        save_txt=True,              
        save_image=True,            
        id2label= id2label,  
        image_paths=valid_paths,
        max_detections=100
)

    print(f"\n✅ Output shape: {results['OUTPUT'].shape}")
    if args.save_txt:
        print(f"📄 Labels saved in: {os.path.join(output_dir, 'labels')}")
    if args.save_image:
        print(f"Images saved in: {output_dir}")

if __name__ == "__main__":
    main()
