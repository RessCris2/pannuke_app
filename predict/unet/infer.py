"""理解 unet 的模型预测输出, 输出应该处理为概率图。
"""
import sys

import torch
from mmseg.apis import MMSegInferencer

sys.path.append("/root/autodl-tmp/pannuke_app/")
# 得到每张图片的预测概率和label， 下一步是利用后处理得到 inst_map.
import numpy as np

from src.evaluation.post_proc import (
    dynamic_watershed_alias,
)  # 是不是应该放在 src.models.unet 中？
from src.evaluation.stats_utils import get_fast_aji_plus, get_fast_pq


def compute_instseg(pred_inst, seg_label):
    """准备计算 mAP 需要的格式。"""
    # pred_inst 为每张图片的 inst_map， seg_label 为每张图片的 label_map，根据每个 inst 对应的label，取数量最大的label为该inst的 type label.
    # 同时, 根据当前最大的label的面积占比，计算类别的概率。
    result = {}
    for inst_id in np.unique(pred_inst):
        inst_mask = pred_inst == inst_id
        inst_label = seg_label[inst_mask]
        inst_type = np.argmax(np.bincount(inst_label))
        score = np.bincount(inst_label).max() / inst_mask.sum()
        # 将 inst_mask 转换为 coco 中的 segmentation 格式
        # TODO: 这里的 inst_mask 需要进行一下处理？
        segmentation = ...
        return inst_type, score, segmentation

    return result


# test initialization
cfg_path = "/root/autodl-tmp/pannuke_app/train/unet/consep/config_consep.py"
ckpt_filename = "/root/autodl-tmp/pannuke_app/train/unet/consep/work-dir/iter_3200.pth"
infer = MMSegInferencer(cfg_path, ckpt_filename, device="cpu")

# 先以一张图片为例，看看输出的结果是什么样的。
img_path = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/imgs/test_1.png"
results = infer([img_path, img_path], return_datasamples=True)
gt_inst_mask = (
    "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/seg_mask/test_1.png"
)

for res in results:
    seg_logits = res.seg_logits.data
    seg_label = res.pred_sem_seg.data
    prob_map = torch.softmax(seg_logits, dim=0).max(axis=0)[0].numpy()
    pred_inst = dynamic_watershed_alias(prob_map, mode="prob")  # 得到每张图的inst_map
    # type_map = compute_type_map(pred_inst, seg_label)

    # 计算 aji 和 pq 只需要 inst_mask 即可？
    aji = get_fast_aji_plus(pred_inst, gt_inst_mask)

    # 计算 map 需要 scores, segmentation, bbox, category_id, image_id


def generate_coco(inst_map, type_map, prob_map):
    pass


# 根据 inst_map, 和label的情况，计算每个inst的label.
