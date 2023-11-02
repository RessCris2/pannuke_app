"""测试加了 aug 之后的数据变换, 先用正常的 run 跑起来，然后用 sweep 调参
"""
import os

os.environ['WANDB_MODE'] = 'offline'
import sys
from argparse import Namespace

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(dir_path)

import time
import warnings

import torch
import wandb
from torch.cuda.amp import autocast

from src.core.utils import get_curtime
from src.seg_unet.train_with_aug import train_profiler as train

warnings.filterwarnings('ignore')


def main(config):
    # wandb.init(project='unet_consep_tune')
    dataset_name='consep'
    model_name = 'seg_unet'
    dir_root = "/root/autodl-tmp/datasets/{}".format(dataset_name)
    log_dir  = "/root/autodl-tmp/train/model_data/{}/{}/{}".format(dataset_name, model_name, get_curtime())

    # train(wandb.config, dataset_name, dir_root, save_dir, )
    num_classes = 5
    train(config, num_classes, model_name, dataset_name, dir_root, log_dir)

if __name__ == "__main__":
    config = Namespace(
        project_name = "unet_demo",
        batch_size = 8,
        epochs = 2,
        lr = 1e-4,
        optimizer = 'adam',
        loss = 'DiceLoss',
        encoder_name = "resnet50",
        activation = "softmax",
        val_interval = 1
    )
    main(config)