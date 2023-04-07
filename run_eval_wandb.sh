#!/bin/bash

/project_ghent/thesis/yolov7/evaluate_yolov7_model.py runs/train/yolov79/weights/best.pt luvdijck/thesis-monocular-depth-estimation/model-3pj640o9:v21  --new-run True
/project_ghent/thesis/yolov7/evaluate_yolov7_model.py runs/train/yolov79/weights/best.pt luvdijck/thesis-monocular-depth-estimation/model-2v64rut6:v30  --new-run True
/project_ghent/thesis/yolov7/evaluate_yolov7_model.py runs/train/yolov79/weights/best.pt luvdijck/thesis-monocular-depth-estimation/model-24zbpoyn:v32  --new-run True
/project_ghent/thesis/yolov7/evaluate_yolov7_model.py runs/train/yolov79/weights/best.pt luvdijck/thesis-monocular-depth-estimation/model-34jz7gv2:v36  --new-run True

# /project_ghent/thesis/keypoint-detection/model_evaluation/evaluate_pnp_model.py ../../yolov5/runs/detect/train13/weights/best.pt "luvdijck/thesis-monocular-depth-estimation/model-34hjaf76:v27"
