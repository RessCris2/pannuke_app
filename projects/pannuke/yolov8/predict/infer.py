import sys

import numpy as np
from ultralytics import YOLO

sys.path.append("/root/autodl-tmp/pannuke_app")
from src.evaluation.stats_utils_v2 import eveluate_one_pic_inst

best_path = (
    "/root/autodl-tmp/pannuke_app/train/ultralytics/runs/segment/train5/weights/best.pt"
)


model = YOLO(best_path)  # load a custom model

# Predict with the model
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# # Run batched inference on a list of images
true_masks_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/inst"
pred_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/imgs"

results = model.predict(
    pred_dir, stream=True, save_dir="pred_data", retina_masks=True
)  # return a generator of Results objects
# Process results generator
metrics = []
basenames = []
for result in results:
    if result.masks is None:
        continue
    path = result.path  # img filename
    basename = path.split("/")[-1].split(".")[0]
    inst_path = f"{true_masks_dir}/{path.split('/')[-1].replace('png', 'npy')}"
    true_inst = np.load(inst_path)
    true_masks = [true_inst == inst_id for inst_id in np.unique(true_inst)[1:]]
    # boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    pred_masks = [mask.data.cpu().numpy().squeeze() for mask in masks]
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs

    metric = eveluate_one_pic_inst(true_masks, pred_masks)
    print(metric)
    basenames.append(basename)
    metrics.append(metric)

import pandas as pd

metrics = np.array(metrics).mean(axis=0)
metrics = pd.DataFrame(metrics, columns=["dice", "aji"])
metrics["basename"] = basenames
print(metrics)
