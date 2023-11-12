import os
import sys

import pytest

sys.path.append("/root/autodl-tmp/pannuke_app/")
from src.data_process.convert2coco import convert_to_coco
from src.data_process.data_transformer import MoNuSAC
from src.data_process.monusac2mask import convert


def monusac2mask():
    """convert monusac to seg_mask(.png) and inst_mask(.npy)
    and convert imgs to png format
    """

    # data_path = "/root/autodl-tmp/pannuke_app/datasets/raw/MoNuSAC/train"
    # dest_path = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/train"
    # convert(data_path, dest_path)

    data_path = "/root/autodl-tmp/pannuke_app/datasets/raw/MoNuSAC/test"
    dest_path = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/test"
    convert(data_path, dest_path)


def convert2coco():
    dataset_name = "monusac"
    data_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/train"
    save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/train/train_annotations.json"
    test_mode = True

    convert_to_coco(dataset_name, data_dir, save_path, test_mode=test_mode)

    dataset_name = "monusac"
    data_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/test"
    save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/test/test_annotations.json"
    # save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/annotations/test_annotations.json"
    test_mode = True

    convert_to_coco(dataset_name, data_dir, save_path, test_mode=test_mode)


def gen_dist_map():
    data_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/train"
    monusac = MoNuSAC(data_dir)
    insts_path = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/train/inst"
    save_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/train/dist_map"
    monusac.gen_dist_map(self, insts_path, save_dir)


if __name__ == "__main__":
    # monusac2mask()
    convert2coco()
    gen_dist_map()
