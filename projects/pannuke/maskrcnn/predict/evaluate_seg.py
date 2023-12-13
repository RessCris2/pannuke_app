"""不做额外的 mAP 计算工作
只评估 dice, aji
"""
import json
import os
import sys

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append("/root/autodl-tmp/pannuke_app")
import glob

import pandas as pd
from src.evaluation.stats_utils_v2 import eveluate_one_pic_inst


def evalute(true_dir, pred_result_dir):
    jsons = glob.glob(f"{pred_result_dir}/*.json")
    metrics = []
    basenames = []
    for json_ in jsons:
        basename = os.path.basename(json_)
        basenames.append(basename)
        with open(json_, "r") as f:
            result = json.load(f)

        # pred_masks = []
        mask = result.pop("masks")
        mask = mask_utils.decode(mask)
        pred_masks = np.transpose(mask, (2, 0, 1))

        true_path = f"{true_dir}/{json_.split('/')[-1].replace('json', 'npy')}"
        inst = np.load(true_path, allow_pickle=True)
        true_masks = [inst == inst_id for inst_id in np.unique(inst)[1:]]
        # 评估 dice, aj

        metric = eveluate_one_pic_inst(true_masks, pred_masks)
        metrics.append(metric)

    metrics = np.array(metrics)
    # metrics = metrics.mean(axis=0)
    metrics = pd.DataFrame(metrics, columns=["dice", "aji"], index=basenames)
    metrics.to_csv(f"{pred_result_dir}/../metrics.csv")
    print(metrics.mean(axis=0))
    print(metrics)


if __name__ == "__main__":
    true_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/inst"
    pred_result_dir = (
        "/root/autodl-tmp/pannuke_app/projects/pannuke/maskrcnn/predict/pred_data/preds"
    )
    evalute(true_dir, pred_result_dir)
