#!/usr/bin/env python3
import json
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from pathlib import Path
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap
from keypoint_detection.utils.load_checkpoints import load_from_checkpoint
from sklearn import metrics
from torchvision.io import read_image
from tqdm import tqdm
import click
import cv2
import torchvision.transforms as T

from yolov7_detect import Yolov7Detector
from cone_detector.metrics import *
from cone_detector.utils import *

import wandb

print_green = lambda x: print("\033[32m" + x + "\033[0m")

# Colours in BGR
CAT_TO_COLOUR = {
    0: (255, 0, 0),
    1: (0, 250, 250),
    2: (0, 100, 220),
    3: (0, 130, 220),
}

# Parameters
MIN_CONE_HEIGHT = 25
METRIC_THRESHOLDS = [0.1, 0.5, 1, 3]
LATENCY_TEST_REPETITIONS = 1000

CONE_IMAGE_MARGIN = 3
IMAGE_SIZE = (128, 96)
image_encoding = T.Resize((int(IMAGE_SIZE[0]), int(IMAGE_SIZE[1])))
device = torch.device("cuda")

# Camera information
W, H = 1920, 1200
Sw, Sh = 9.21, 5.76
focal_length = 8

m = np.load("../keypoint-detection/model_evaluation/camera_calibration_baumer.npz")
camera_matrix = m["camera_matrix"]
distortion_matrix = m["distortion_matrix"]


@click.command()
@click.argument(
    "yolo_model_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.argument(
    "artifact_id",
    type=str,
)
@click.option(
    '--new-run',
    default=False,
    help='Create a new W&B run instead of overwriting the old one.'
)
def main(yolo_model_path, artifact_id, new_run):
    # Connect to wandb run
    print_green("Retrieving wandb run")
    run = wandb.init(
        project="thesis-monocular-depth-estimation",
        id=artifact_id[49:57],
        resume="must",
    )

    artifact = run.use_artifact(artifact_id, type="model")
    artifact_dir = artifact.download()
    checkpoint_path = Path(artifact_dir) / "model.ckpt"
    kp_model = load_from_checkpoint(checkpoint_path)
    detector = Yolov7Detector(yolo_model_path)
    
    if new_run:
        run.finish()
        run = wandb.init(
            project="thesis-monocular-depth-estimation",\
        )

    # Initialise the dataset
    dataset_path = Path("../test_set_real/dataset.json")

    # Open the dataset and map image ids to their annotations
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    ann_by_image_id = {}

    for annot in dataset["annotations"]:
        if annot["image_id"] not in ann_by_image_id:
            ann_by_image_id[annot["image_id"]] = [annot]
        else:
            ann_by_image_id[annot["image_id"]].append(annot)

    # Predict the data
    global_predictions, global_gt = [], []
    prediction_times = []
    image_ids = []

    print_green("Predicting test set cones")
    # Form predictions for the entire dataset
    for image_obj in tqdm(dataset["images"], desc="Predicting cone locations"):
        # Load in the image
        original_image = (
            read_image(str(dataset_path.parent / image_obj["file_name"])).float() / 255
        )
        vis_img = cv2.imread(str(dataset_path.parent / image_obj["file_name"]))
        image_ids.append(image_obj['id'])

        # Predict keypoints
        yolo_time, detections = detector.detect(vis_img)
        prediction_times.append(
            yolo_time
        )  # Firstly append the YOLO time, the cone extraction will be added in-place later

        predictions = np.empty((0, 4))
        for detection in detections:
            category = int(detection[-1])
            bbox_kpts = [round(x) for x in detection[:4]]
            bbox_kpts[0] = max(0, bbox_kpts[0] - CONE_IMAGE_MARGIN)
            bbox_kpts[1] = max(0, bbox_kpts[1] - CONE_IMAGE_MARGIN)
            
            vis_img = cv2.rectangle(vis_img, bbox_kpts[:2], bbox_kpts[2:], CAT_TO_COLOUR[category], 2)

            cone_img = original_image[
                :, bbox_kpts[1] : bbox_kpts[3], bbox_kpts[0] : bbox_kpts[2]
            ]

            original_size = cone_img.shape[1:]
            with torch.no_grad():
                cone_img = image_encoding(
                    cone_img
                )  # Don't forget to rescale the image before inference
                cone_img = cone_img[np.newaxis, ...]
                result = kp_model(cone_img)

            tic = time.perf_counter()

            # List with elements: Tensor of #detectionsx2
            kpts = []
            for i in range(2):
                tens = torch.Tensor(get_keypoints_from_heatmap(result[0, i], 2))
                # Don't forget to undo any image resizing afterwards
                if len(tens) != 0:
                    tens[:, 0] = (
                        tens[:, 0] * original_size[1] / IMAGE_SIZE[1] + bbox_kpts[0]
                    )
                    tens[:, 1] = (
                        tens[:, 1] * original_size[0] / IMAGE_SIZE[0] + bbox_kpts[1]
                    )
                kpts.append(tens)

            # Without a bottom kpts -> no detection
            if len(kpts[1]) == 0:
                continue
            # Without a top keypoint -> guess it
            if len(kpts[0]) == 0:
                kpts[0] = torch.Tensor(
                    [
                        [
                            bbox_kpts[0] + CONE_IMAGE_MARGIN + original_size[1] / 2,
                            bbox_kpts[1] + CONE_IMAGE_MARGIN,
                        ]
                    ]
                )
            # If there are multiple bottoms, take the left one
            if kpts[1].shape[0] > 1:
                if kpts[1][0, 0] < kpts[1][1, 0]:
                    kpts[1] = kpts[1][0].view(1, 2)
                else:
                    kpts[1] = kpts[1][1].view(1, 2)
            
            for pred in kpts:
                pred_int = pred[0].int().numpy()
                pred_int = (pred_int[0], pred_int[1])
                vis_img = cv2.circle(
                    vis_img, pred_int, 3, CAT_TO_COLOUR[category], -1
                )
                vis_img = cv2.circle(vis_img, pred_int, 2, (0, 0, 0), -1)
                vis_img = cv2.circle(vis_img, pred_int, 1, (255, 255, 255), -1)

            cone_height = kpts[1][0, 1] - kpts[0][0, 1]
            cone_pos = height_to_pos(
                cone_height, kpts[1], category, camera_matrix, focal_length, Sh, H
            )[0].numpy()
            cone = np.array([category, *cone_pos])
            predictions = np.vstack((predictions, cone))

            toc = time.perf_counter()
            prediction_times[-1] += toc - tic

        # If there is GT data available, extract the GT positions
        if image_obj["id"] not in ann_by_image_id:
            gt = None
        else:
            gt = extract_positions(ann_by_image_id[image_obj["id"]], MIN_CONE_HEIGHT)

        if len(predictions) == 0:
            predictions = None
        pred_count = 0 if predictions is None else predictions.shape[0]
        gt_count = 0 if gt is None else gt.shape[0]
        print(f"#detected/#gt -> {pred_count}/{gt_count}")

        global_predictions.append(predictions)
        global_gt.append(gt)
        
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        wandb.log({f"[{image_obj['id']}] Detections": wandb.Image(vis_img)})

    print_green("Calculate metrics")
    columns = []
    for threshold in METRIC_THRESHOLDS:
        columns += [
            f"lucas_{threshold}",
            f"colour_{threshold}",
            f"matching_{threshold}",
            f"pos_2d_{threshold}",
            f"pos_3d_{threshold}",
        ]
    df = pd.DataFrame(columns=columns)
    for i in range(len(global_predictions)):
        entry = []
        for threshold in METRIC_THRESHOLDS:
            pred_to_gt, gt_fn = match_cones(
                global_gt[i], global_predictions[i], threshold
            )

            l_score = lucas_score(
                global_gt[i],
                global_predictions[i],
                pred_to_gt,
                gt_fn,
                threshold,
            )
            l2_score = lucas_score2(
                global_gt[i],
                global_predictions[i],
                pred_to_gt,
                gt_fn,
                threshold,
            )
            c_score = colour_score(
                global_gt[i],
                global_predictions[i],
                pred_to_gt,
                gt_fn,
            )
            m_score = matching_score(
                global_gt[i],
                global_predictions[i],
                pred_to_gt,
                gt_fn,
            )
            score_2d, score_3d = pos_score(
                global_gt[i], global_predictions[i], pred_to_gt, gt_fn
            )
            if threshold == 0.5:
                fig = generate_plot(
                    global_predictions[i],
                    global_gt[i],
                    f"[{image_ids[i]}] Lucas score 0.5: {round(l_score)}/{l2_score:.2f} ({c_score:.2f}, {m_score:.2f}, {score_2d:.2f})",
                )
                wandb.log({f"lucas_score_0.5_{i}": wandb.Image(fig)})

            entry += [
                l_score,
                colour_score(
                    global_gt[i],
                    global_predictions[i],
                    pred_to_gt,
                    gt_fn,
                ),
                matching_score(
                    global_gt[i],
                    global_predictions[i],
                    pred_to_gt,
                    gt_fn,
                ),
                score_2d,
                score_3d,
            ]

        df.loc[len(df)] = entry
    print("Median scores:")
    print(df.median())

    # Log metrics
    table = wandb.Table(data=df, columns=columns)
    for threshold in METRIC_THRESHOLDS:
        wandb.log(
            {
                f"test/metrics/lucas/{threshold}_histogram": wandb.plot.histogram(
                    table,
                    f"lucas_{threshold}",
                    title=f"Lucas score (threshold {threshold})",
                )
            }
        )

    for col_name in columns:
        wandb.run.summary[f"test/metrics/median_{col_name}"] = df[col_name].median()

    # Finally perform a speed test
    print_green("Latency check")
    timings = analyse_model_speed(kp_model, IMAGE_SIZE, LATENCY_TEST_REPETITIONS)
    data = [[s] for s in timings]
    table = wandb.Table(data=data, columns=["timings"])
    wandb.log(
        {
            "latency_histogram": wandb.plot.histogram(
                table, "timings", title="Model latency"
            )
        }
    )
    wandb.run.summary["median_model_latency"] = np.median(timings)
    wandb.run.summary["median_total_latency"] = np.median(timings) + np.median(
        prediction_times
    )
    print(f"Median model latency: {wandb.run.summary['median_model_latency']} ms")
    print(f"Median total latency: {wandb.run.summary['median_total_latency']} ms")

    run.finish()


if __name__ == "__main__":
    main()
