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
from src.data_process.pycococreatortools import binary_mask_to_polygon
from src.evaluation.evaluate import calculate_map, evaluate
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
        inst_mask = np.where(inst_map == int(inst_uid), 1, 0)
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
        item.update({"segmentation": binary_mask_to_polygon(inst_mask)})
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


def evaluate_seg(true_dir, pred_dir):
    """评估一个文件夹下的所有图片"""
    true_paths = glob.glob(os.path.join(true_dir, "*.npy"))
    # pred_paths = glob.glob(os.path.join(pred_dir, "*.mat"))

    metrics = []
    basenames = []
    for true_path in true_paths:
        basename = os.path.basename(true_path).replace(".npy", ".png")
        pred_path = os.path.join(pred_dir, basename.replace(".png", ".mat"))
        try:
            true_masks = convert_inst2masks(true_path)
            pred_masks = convert_mat2masks(pred_path)  # 预测的数据里可能会没有instance
        except ValueError:
            continue
        basenames.append(basename)
        metric = eveluate_one_pic_inst(true_masks, pred_masks)
        metrics.append(metric)

    metrics_pd = pd.DataFrame(metrics, index=basenames, columns=["dice", "aji"])
    # .to_csv("hovernet.csv")
    avg_metric = np.mean(metrics, axis=0)
    return avg_metric, metrics_pd


def evaluate_map(ann_file, pred_dir):
    """计算所有图片的 map"""
    coco_api = COCO(ann_file)

    # for each image_id, get the corresponding file_name
    results = []
    for image in coco_api.dataset["images"]:
        file_name = "{}/{}".format(pred_dir, image["file_name"].replace("png", "mat"))
        items = convert_mat2coco(file_name, image["id"])
        results.extend(items)

    overall_map, map_pd = calculate_map(results, coco_api)
    return overall_map, map_pd


# def evaluate_hover(ann_file, pred_result_dir, true_dir, save_path=None):
#     avg_metric, metrics_pd = evaluate_seg(true_dir, pred_result_dir)
#     overall_map, map_pd = calculate_map(ann_file, pred_result_dir)

#     res = pd.merge(metrics_pd, map_pd, left_index=True, right_index=True)
#     res["average_dice"] = avg_metric[0]
#     res["average_aji"] = avg_metric[1]
#     res["average_map"] = overall_map[0]
#     res["average_map50"] = overall_map[1]

#     if save_path is not None:
#         res.to_csv(save_path)

#     return res


if __name__ == "__main__":
    pred_dir = (
        "/root/autodl-tmp/pannuke_app/projects/pannuke/hovernet/predict/pred_data/mat"
    )
    true_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/inst"
    ann_file = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/test_annotations.json"
    # calculate_map(ann_file, pred_dir)
    save_path = (
        "/root/autodl-tmp/pannuke_app/projects/pannuke/hovernet/evaluation/hovernet.csv"
    )
    evaluate(evaluate_seg, evaluate_map, ann_file, pred_dir, true_dir, save_path)
