"""
train model DIST.
"""
import os
import sys

# file_path = os.path.abspath(__file__)
# dir_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append("/root/autodl-tmp/viax")

from argparse import Namespace

import torch
import wandb

from src.core.utils import get_curtime
from src.distance.dist_net import DIST
from src.distance.train import train_wb as dist_train


# 调参后一次性设定
def main():
    # config = Namespace(
    #     project_name = "unet_demo",
    #     batch_size = 8,
    #     epochs = 100,
    #     lr = 1e-3,
    #     weight_decay=0.0001,
    #     # optimizer = 'adam',
    #     # loss = 'DiceLoss', # "JaccardLoss", "DiceLoss", "TverskyLoss", "FocalLoss", "LovaszLoss", "SoftBCEWithLogitsLoss"
    #     # encoder_name = "resnet50",
    #     # activation = "softmax",
    #     val_interval = 1,
    #     save_interval = 10,
    #     num_workers= 8,
    #     num_features=6

    # )
    dataset_name = 'pannuke'
    config = Namespace(
        project_name=f"dist_{dataset_name}_demo",
        batch_size=8,
        epochs=100,
        lr=1e-3,
        weight_decay=0.0001,
        # optimizer = 'adam',
        # loss = 'DiceLoss', # "JaccardLoss", "DiceLoss", "TverskyLoss", "FocalLoss", "LovaszLoss", "SoftBCEWithLogitsLoss"
        # encoder_name = "resnet50",
        # activation = "softmax",
        val_interval=1,
        save_interval=10,
        num_workers=8,
        num_features=6)

    wandb.init(config=config.__dict__, project=config.project_name)
    #     wandb.init(project='unet_consep_tune')

    dataset_name = 'pannuke'
    model_name = 'dist'
    dir_root = "/root/autodl-tmp/viax/datasets/{}".format(dataset_name)
    log_dir = "/root/autodl-tmp/viax/train/model_data/{}/{}/{}".format(
        dataset_name, model_name, get_curtime())

    # train(wandb.config, dataset_name, dir_root, save_dir, )
    num_classes = 6
    dist_train(wandb.config, dataset_name, dir_root, log_dir)


if __name__ == "__main__":
    main()
