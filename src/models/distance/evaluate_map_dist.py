import sys

sys.path.append("/root/autodl-tmp/archive/v2/models/dist")
sys.path.append("/root/autodl-tmp/archive/metrics")
import pathlib
from os.path import join as opj

import cv2
import numpy as np
import torch
from dataloader import data_aug
from dist_net import DIST
from post_proc import post_process
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from utils import (find_files, get_bounding_box, load_img, load_model,
                   rm_n_mkdir)

from predict import predict


def transfer_inst_format(true_inst):
    """ 将 inst [256, 256]
        转换为
        dict(
        scores:
        labels:
        bboxes:
        masks: (N, 256, 256)
        )
        的格式
    """
    true = {}
    true_inst_id = np.unique(true_inst)[1:]
    true_masks = np.stack([(true_inst==inst_id).astype(int) for inst_id  in true_inst_id ])
    true_bboxes = np.stack([get_bounding_box((true_inst==inst_id).astype(int)) for inst_id  in true_inst_id ])
    true_scores = [0.99] * len(true_inst_id)
    # fake one!
    true_labels = [1] * len(true_inst_id)
    
    true.update({'bboxes': true_bboxes})
    true.update({'scores': true_scores})
    true.update({'masks': true_masks})
    true.update({'labels': true_labels})
    return true

save_path = "/root/autodl-tmp/archive/v2/model_data/dist/pannuke/202305301616/202305301622/202305301627/202305301632/202305301637/202305301643/202305301648/epoch_30.pth"
model = DIST(num_features=6)
model = load_model(model, save_path)

# 只对一张图片进行评估
img = load_img("/root/autodl-tmp/archive/datasets/pannuke/patched/coco_format/images/val/0.jpg")
pred = predict(model, img) # (256, 256)
img_path = "/root/autodl-tmp/archive/datasets/pannuke/patched/coco_format/images/val/0.jpg"
true_inst = load_img(img_path.replace("images", "inst").replace("jpg", "npy"))
pred_inst = transfer_inst_format(pred)
true_inst = transfer_inst_format(true_inst)

preds = [dict(
        boxes=torch.tensor(pred_inst['bboxes']),
        scores=torch.tensor(pred_inst['scores']),
        labels=torch.tensor(pred_inst['labels']),
        masks = torch.tensor(pred_inst['masks'], dtype=torch.uint8)
)]

target = [dict(
        boxes=torch.tensor(true_inst['bboxes']),
        labels=torch.tensor(true_inst['labels']),
        masks = torch.tensor(true_inst['masks'], dtype=torch.uint8)
)]

metric = MeanAveragePrecision(iou_type='segm')
metric.update(preds, target)
from pprint import pprint

#     pprint()
metric = metric.compute()
pprint(metric)