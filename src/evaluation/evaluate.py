import glob
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# def convert_binary2polygon(binary_mask):

#     return polygon


def calculate_map(results, coco_api):
    pred = coco_api.loadRes(results)
    coco_eval = COCOeval(coco_api, pred, iouType="segm")
    # coco_eval.params.maxDets = [100, 500, 1000]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    overall_map = coco_eval.stats[:2]  # 0 是 map, 1 是map50

    per_image_mAPs = []
    basenames = []
    for image in coco_api.dataset["images"]:
        # for img_id in coco_api.getImgIds():
        basename = image["file_name"]
        img_id = image["id"]
        coco_eval.params.imgIds = [img_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # 获取每张图像的 mAP 值
        per_image_mAPs.append(coco_eval.stats[:2])
        basenames.append(basename)

    map_pd = pd.DataFrame(per_image_mAPs, index=basenames)

    # 打印每张图像的 mAP 值
    # for i, mAP in enumerate(per_image_mAPs):
    #     print(f"mAP for image {i + 1}: {mAP}")
    return overall_map, map_pd


def evaluate(
    evaluate_seg,
    evaluate_map,
    ann_file,
    pred_dir,
    true_dir,
    save_path=None,
    model=None,
):
    avg_metric, metrics_pd = evaluate_seg(true_dir, pred_dir)
    overall_map, map_pd = evaluate_map(ann_file=ann_file, pred_dir=pred_dir)

    res = pd.merge(metrics_pd, map_pd, left_index=True, right_index=True)
    res["average_dice"] = avg_metric[0]
    res["average_aji"] = avg_metric[1]
    res["average_map"] = overall_map[0]
    res["average_map50"] = overall_map[1]

    if save_path is not None:
        res.to_csv(save_path)

    return res
