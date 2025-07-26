#!/bin/bash
python test_trism_cv.py \
    --model_name yolov_deyo_ensemble \
    --url localhost:8001 \
    --data /home/nhattan05022003/coding/Tien_project/Triton_Thanh/odlab-triton/assets \
    --output results \
    --auto_config \
    --batch_size 3 \
    --save-txt \
    --save-image \
    

  # python test_trism_cv.py \
#   --model_name yolov_deyo_ensemble \
#   --save-txt \
#   --save-image \
#   --data /home/nhattan05022003/coding/Tien_project/Triton_Thanh/odlab-triton/assets \
#   --label-file  /home/nhattan05022003/coding/Tien_project/Triton_Thanh/odlab-triton/src/labels.txt \
#   --max-detections 100 \
#   --output result \
#   --auto-config