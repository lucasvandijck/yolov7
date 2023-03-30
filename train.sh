#!/bin/bash

cd /project_ghent/thesis/yolov7

pip install -r requirements.txt

wandb login d630d6ebc6e28ad0a4151127a43866ed9a92eba0

python train.py --workers 8 --device 0 --batch-size $1 --data data/cones.yaml --img 640 640 --cfg cfg/training/yolov7x.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml