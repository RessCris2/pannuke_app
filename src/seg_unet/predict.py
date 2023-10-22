import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import glob

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
from post_proc import post_process
import sys
sys.path.append("/root/autodl-tmp/archive")
from metrics.compute_inst import run_one_inst_stat

# model = torch.load(path,  map_location='cpu')
def predict(model_path, img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))
    x_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    model = torch.load(model_path, map_location='cpu')
    res = model.predict(x_tensor).squeeze().detach().numpy()
    prob_map = res[1:].sum(axis=0)
    pred = post_process(prob_map, mode='prob')
    return res, prob_map, pred # [6, 256, 256] 的结果


def evaluate_one_pic(model_path, img_path):
    """如何把这个[6, 256, 256] 的结果转换为 inst_id 的结果？
    """
    path = "/root/autodl-tmp/datasets/pannuke/inst/test/0.npy"
    true = np.load(path)

    # img_path = "/root/autodl-tmp/datasets/pannuke/coco_format/images/test/0.jpg"
    _, prob_map, pred = predict(model_path, img_path)
    metrics = run_one_inst_stat(true, pred, match_iou=0.5)
    return metrics

if __name__ == '__main__':
    model_path = "/root/autodl-tmp/archive/pannuke_v2/unet_seg/model/best_model.pth"
    img_path = "/root/autodl-tmp/datasets/pannuke/coco_format/images/test/0.jpg"
    evaluate_one_pic(model_path, img_path)
    print('xxx')