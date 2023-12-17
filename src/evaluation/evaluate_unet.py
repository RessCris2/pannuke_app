"""理解 unet 的模型预测输出, 输出应该处理为概率图。
"""
import json
import os
import pathlib
import sys

import torch
from mmseg.apis import MMSegInferencer
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append("/root/autodl-tmp/pannuke_app/")
# 得到每张图片的预测概率和label， 下一步是利用后处理得到 inst_map.
import numpy as np
import pandas as pd
from src.evaluation.evaluate import calculate_map, evaluate
from src.evaluation.post_proc import (
    dynamic_watershed_alias,
)  # 是不是应该放在 src.models.unet 中？
from src.evaluation.stats_utils import (
    get_dice_1,
    get_fast_aji_plus,
    get_fast_pq,
    remap_label,
)


def compute_seg(true, pred):
    metrics = []
    # to ensure that the instance numbering is contiguous
    pred = remap_label(pred, by_size=False)
    true = remap_label(true, by_size=False)

    # pq_info = get_fast_pq(true, pred, match_iou=0.5)[0]
    metrics.append(get_dice_1(true, pred))
    # metrics.append(pq_info[0])  # dq
    metrics.append(get_fast_aji_plus(true, pred))
    return metrics


def find_img_id(file_name, coco_api):
    """根据文件名找到对应的 img_id"""
    image_id = None
    for img_id, img_info in coco_api.imgs.items():
        if img_info["file_name"] == file_name:
            image_id = img_id
            break
    return image_id


def compute_instseg(coco_api, pred_inst, seg_label, img_path):
    """准备计算 mAP 需要的格式。"""
    # pred_inst 为每张图片的 inst_map， seg_label 为每张图片的 label_map，根据每个 inst 对应的label，取数量最大的label为该inst的 type label.
    # 同时, 根据当前最大的label的面积占比，计算类别的概率。
    """
    for d1, d2, d3, d4 in zip(
            result["bbox"],
            result["category_id"],
            result["segmentation"],
            result["score"],
        ):
            item = {}
            item.update({"bbox": d1})
            item.update({"category_id": d2})
            item.update({"segmentation": d3})
            item.update({"score": d4})
            item.update({"image_id": image["id"]})
    """
    img_id = find_img_id(os.path.basename(img_path), coco_api)
    items = []

    for inst_id in np.unique(pred_inst)[1:]:
        inst_mask = pred_inst == inst_id
        inst_label = seg_label[inst_mask]
        inst_type = np.argmax(np.bincount(inst_label))
        score = np.bincount(inst_label).max() / inst_mask.sum()
        # 将 inst_mask 转换为 coco 中的 segmentation 格式
        # TODO: 这里的 inst_mask 需要进行一下处理？
        segmentation = mask_utils.encode(np.asfortranarray(inst_mask.astype(np.uint8)))
        item = {}
        # item.update({"bbox": d1})  # 是否需要输出 bbox?
        item.update({"category_id": int(inst_type)})
        item.update({"segmentation": segmentation})
        item.update({"score": score})
        item.update({"image_id": img_id})
        # return item
        items.append(item)

    return items


def evaluate_unet(
    cfg_path, ckpt_filename, pred_dir, ann_file, mid_save_dir, save_path
):  # (pred_dir, cfg_path, ckpt_path, device="cuda"):
    """
    infer 一个文件夹，并且计算了相应指标（取均值）
    """
    # test initializations
    # cfg_path = (
    #     "/root/autodl-tmp/pannuke_app/projects/consep/unet/train/config_consep.py"
    # )

    # ckpt_filename = (
    #     "/root/autodl-tmp/pannuke_app/projects/consep/unet/train/work-dir/iter_400.pth"
    # )
    infer = MMSegInferencer(cfg_path, ckpt_filename, device="cuda")

    # pred_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/imgs"
    # ann_file = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/test_annotations.json"
    results = infer(pred_dir, return_datasamples=True)

    os.makedirs(mid_save_dir, exist_ok=True)

    coco_api = COCO(ann_file)
    metrics = []
    map_data = []

    basenames = []
    for res in results:
        img_path = res.metainfo["img_path"]
        basename = os.path.basename(img_path)
        basenames.append(basename)
        true_inst = np.load(img_path.replace("imgs", "inst").replace("png", "npy"))
        seg_logits = res.seg_logits.data
        seg_label = res.pred_sem_seg.data
        try:
            prob_map = torch.softmax(seg_logits, dim=0).max(axis=0)[0].cpu().numpy()
        except Exception as e:
            prob_map = torch.softmax(seg_logits, dim=0).max(axis=0)[0].numpy()

        pred_inst = dynamic_watershed_alias(prob_map, mode="prob")  # 得到每张图的inst_map
        # inst_path = f"{mid_save_dir}/ {basename.replace('png', 'npy')}"
        # pred_path = inst_path.replace("inst", "prob_map")
        # np.save(pred_path, prob_map)
        # np.save(save_path, inst_path)
        metric = compute_seg(true_inst, pred_inst)
        metrics.append(metric)
        seg_label = seg_label.cpu().numpy().squeeze()
        ann = compute_instseg(coco_api, pred_inst, seg_label, img_path)
        map_data.extend(ann)

    metrics_pd = pd.DataFrame(metrics, columns=["dice", "aji"], index=basenames)
    # print(metrics.mean(axis=0))
    avg_metric = metrics_pd.mean(skipna=True).values

    overall_map, map_pd = calculate_map(map_data, coco_api)
    res = pd.merge(metrics_pd, map_pd, left_index=True, right_index=True)
    res["average_dice"] = avg_metric[0]
    res["average_aji"] = avg_metric[1]
    res["average_map"] = overall_map[0]
    res["average_map50"] = overall_map[1]

    if save_path is not None:
        res.to_csv(save_path)

    return res


if __name__ == "__main__":
    pred_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/imgs"
    cfg_path = (
        "/root/autodl-tmp/pannuke_app/projects/pannuke/unet/train/work-dir/config.py"
    )
    ckpt_filename = "/root/autodl-tmp/pannuke_app/projects/pannuke/unet/train/work-dir/iter_32000.pth"
    ann_file = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/test_annotations.json"
    mid_save_dir = "/root/autodl-tmp/pannuke_app/projects/pannuke/unet/predict/inst/"
    save_path = "/root/autodl-tmp/pannuke_app/projects/pannuke/unet/predict/unet.csv"
    res = evaluate_unet(
        cfg_path, ckpt_filename, pred_dir, ann_file, mid_save_dir, save_path
    )
    print(res)
