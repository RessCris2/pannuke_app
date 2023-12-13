import glob
import json
import os
import pathlib
import sys

import numpy as np
import scipy.io as sio
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append("/root/autodl-tmp/pannuke_app/")
import pandas as pd
from src.evaluation.stats_utils_v2 import eveluate_one_pic_inst
from src.models.hover.compute_stats import run_nuclei_inst_stat

# metrics = run_nuclei_inst_stat(pred_dir, true_dir, print_img_stats=False)
# print(metrics)

# 遇到没有任何匹配的情况， aji 的计算会报错。
# 但是这种情况下，不应该报错，而是应该返回 0


def convert_mat2coco(mat_path, img_id):
    """将数据转换为 coco 的格式"""
    label = sio.loadmat(mat_path)
    json_path = mat_path.replace("mat", "json")
    with open(json_path, "r") as f:
        json_label = json.load(f)["nuc"]

    inst_map = label["inst_map"]
    items = []
    for inst_uid in json_label.keys():
        inst_mask = inst_map == inst_uid
        category_id = json_label[inst_uid]["type"]
        score = json_label[inst_uid]["type_prob"]
        # inst_bbox = np.array()
        # 改为 x,y,w,h

        [[rmin, cmin], [rmax, cmax]] = json_label[inst_uid]["bbox"]
        w = cmax - cmin
        h = rmax - rmin
        bbox = [cmin, rmin, w, h]

        item = {}
        item.update({"bbox": bbox})
        item.update({"category_id": category_id})
        item.update({"segmentation": inst_mask})
        item.update({"score": score})
        item.update({"image_id": img_id})
        items.append(item)
    return items


# 将数据都转换为 pred_masks, true_masks 处理？
def convert_inst2masks(inst_path):
    """将 inst [H, W] 转换为 masks [N, H, W]"""
    inst = np.load(inst_path, allow_pickle=True)
    inst = inst.astype(np.int32)
    inst_ids = np.unique(inst)[1:]
    masks = []
    for inst_id in inst_ids:
        inst_mask = inst == inst_id
        masks.append(inst_mask)
    return np.stack(masks)


def convert_mat2masks(mat_path):
    """将 inst [H, W] 转换为 masks [N, H, W]"""
    inst = sio.loadmat(mat_path)["inst_map"]
    inst = inst.astype(np.int32)
    inst_ids = np.unique(inst)[1:]
    masks = []
    if len(inst_ids) == 0:
        raise ValueError("No instance in mat file.")
    for inst_id in inst_ids:
        inst_mask = inst == inst_id
        masks.append(inst_mask)
    return np.stack(masks)


def evaluate_pq(true_dir, pred_dir):
    """评估一个文件夹下的所有图片"""
    true_paths = glob.glob(os.path.join(true_dir, "*.npy"))
    # pred_paths = glob.glob(os.path.join(pred_dir, "*.mat"))

    metrics = []
    basenames = []
    for true_path in true_paths:
        basename = pathlib.Path(true_path).stem

        pred_path = os.path.join(pred_dir, basename + ".mat")
        try:
            true_masks = convert_inst2masks(true_path)
            pred_masks = convert_mat2masks(pred_path)  # 预测的数据里可能会没有instance
        except ValueError:
            continue
        metric = eveluate_one_pic_inst(true_masks, pred_masks)
        if metric[0] > 1 or metric[1] > 1:
            print(basename)
            continue
        print(basename, metric)
        metrics.append(metric)
        basenames.append(basename)
    metrics = pd.DataFrame(metrics, columns=["dice", "aji"], index=basenames)
    metrics.to_csv(f"{pred_dir}/../metrics.csv")
    print(metrics.mean(axis=0))
    print(metrics)
    avg_metric = np.mean(metrics, axis=0)
    return avg_metric


def calculate_map(ann_file, pred_result_dir):
    """计算所有图片的 map"""
    coco_api = COCO(ann_file)

    # for each image_id, get the corresponding file_name
    results = []
    for image in coco_api.dataset["images"]:
        file_name = "{}/{}".format(
            pred_result_dir, image["file_name"].replace("png", "mat")
        )
        items = convert_mat2coco(file_name, image["id"])
        results.extend(items)
    pred = coco_api.loadRes(results)
    coco_eval = COCOeval(coco_api, pred, iouType="bbox")
    # coco_eval.params.maxDets = [100, 500, 1000]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    pred_dir = "pred_data/mat"
    true_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/inst"
    # metrics = evaluate_pq(true_dir, pred_dir)
    # print(metrics)
    # mat_path = "/root/autodl-tmp/pannuke_app/projects/consep/hovernet/predict/pred_data/mat/test_12.mat"
    # convert_mat2coco(mat_path)
    ann_file = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/test_annotations.json"
    # calculate_map(ann_file, pred_dir)
    evaluate_pq(true_dir, pred_dir)
    # metrics = run_nuclei_inst_stat(pred_dir, true_dir, print_img_stats=False)
    # print(metrics)
