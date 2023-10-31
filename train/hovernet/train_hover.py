"""run_train.py

Main HoVer-Net training script.

Usage:
  run_train.py [--gpu=<id>] [--view=<dset>]
  run_train.py (-h | --help)
  run_train.py --version

Options:
  -h --help       Show this string.
  --version       Show version.
  --gpu=<id>      Comma separated GPU list. [default: 0,1,2,3]
  --view=<dset>   Visualise images after augmentation. Choose 'train' or 'valid'.
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import random

import cv2

cv2.setNumThreads(0)
import argparse
import glob
import importlib
import inspect
import json
import shutil

import numpy as np
import torch
from docopt import docopt
from tensorboardX import SummaryWriter
from torch.nn import DataParallel  # TODO: switch to DistributedDataParallel
from torch.utils.data import DataLoader

torch.cuda.empty_cache()

import sys
import time

import cv2
# from dataset import get_dataset
import torch.optim as optim

# file_path = os.path.abspath(__file__)
# dir_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append('/root/autodl-tmp/viax')
import wandb

from src.hovernet.run_train import TrainManager

# def main():
#     wandb.init(project='consep_hover')
#     dataset_name = 'consep'
#     type_classificaion=True
#     nr_type = 5
#     pretrained = '/root/autodl-tmp/viax/src/hovernet/pretrained/ImageNet-ResNet50-Preact_pytorch.tar'
#     # pretrained = "/root/autodl-tmp/archive/v2/model_data/cpm17/202305242347/01/net_epoch=37.tar"
#     train_dir_list = f"/root/autodl-tmp/viax/datasets/{dataset_name}/images/train"
#     valid_dir_list = f"/root/autodl-tmp/viax/datasets/{dataset_name}/images/test"
#     log_dir = f"/root/autodl-tmp/viax/train/model_data/{dataset_name}/hovernet"
#     trainer = TrainManager(dataset_name, nr_type, train_dir_list, valid_dir_list, log_dir, pretrained)
#     trainer.run()
#     wandb.finish()
  
# def main():
#     wandb.init(project='monusac_hover')
#     dataset_name = 'monusac'
#     type_classificaion=True
#     nr_type = 5
#     pretrained = '/root/autodl-tmp/viax/src/hovernet/pretrained/ImageNet-ResNet50-Preact_pytorch.tar'
#     # pretrained = "/root/autodl-tmp/archive/v2/model_data/cpm17/202305242347/01/net_epoch=37.tar"
#     train_dir_list = f"/root/autodl-tmp/viax/datasets/{dataset_name}/images/train"
#     valid_dir_list = f"/root/autodl-tmp/viax/datasets/{dataset_name}/images/test"
#     log_dir = f"/root/autodl-tmp/viax/train/model_data/{dataset_name}/hovernet"
#     trainer = TrainManager(dataset_name, nr_type, train_dir_list, valid_dir_list, log_dir, pretrained)
#     trainer.run()
#     wandb.finish()

def main():
    wandb.init(project='pannuke_hover')
    dataset_name = 'pannuke'
    type_classificaion=True
    nr_type = 6
    pretrained = '/root/autodl-tmp/viax/src/hovernet/pretrained/ImageNet-ResNet50-Preact_pytorch.tar'
    # pretrained = "/root/autodl-tmp/archive/v2/model_data/cpm17/202305242347/01/net_epoch=37.tar"
    train_dir_list = f"/root/autodl-tmp/viax/datasets/{dataset_name}/images/train"
    valid_dir_list = f"/root/autodl-tmp/viax/datasets/{dataset_name}/images/test"
    log_dir = f"/root/autodl-tmp/viax/train/model_data/{dataset_name}/hovernet"
    trainer = TrainManager(dataset_name, nr_type, train_dir_list, valid_dir_list, log_dir, pretrained)
    trainer.run()
    wandb.finish()
    


if __name__ == "__main__":
    main()
