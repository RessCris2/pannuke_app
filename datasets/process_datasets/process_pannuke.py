import os
import sys

import pytest

sys.path.append("/root/autodl-tmp/pannuke_app/")
from src.data_process.convert2coco import convert_to_coco
from src.data_process.data_transformer import PanNuke
from src.data_process.pannuke2mask import pn2img, pn2inst_type


def pannuke2mask():
    """convert PanNuke to seg_mask(.png) and inst_mask(.npy)
    and convert imgs to png format
    """
    # TODO: finish the path description
    data_path = "/root/autodl-tmp/pannuke_app/datasets/raw/PanNuke/fold1/Fold 1/images/fold1/images.npy"
    dest_path = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/train"
    prefix = "train"
    pn2img(data_path, dest_path, prefix)

    data_path = "/root/autodl-tmp/pannuke_app/datasets/raw/PanNuke/fold2/Fold 2/images/fold2/images.npy"
    dest_path = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test"
    prefix = "test"
    pn2img(data_path, dest_path, prefix)

    data_path = "/root/autodl-tmp/pannuke_app/datasets/raw/PanNuke/fold3/Fold 3/images/fold3/images.npy"
    dest_path = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/val"
    prefix = "val"
    pn2img(data_path, dest_path, prefix)

    # transform masks to inst_mask and seg_mask
    data_path = "/root/autodl-tmp/pannuke_app/datasets/raw/PanNuke/fold1/Fold 1/masks/fold1/masks.npy"
    dest_path = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/train"
    prefix = "train"
    pn2inst_type(data_path, dest_path, prefix)

    data_path = "/root/autodl-tmp/pannuke_app/datasets/raw/PanNuke/fold2/Fold 2/masks/fold2/masks.npy"
    dest_path = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test"
    prefix = "test"
    pn2inst_type(data_path, dest_path, prefix)

    data_path = "/root/autodl-tmp/pannuke_app/datasets/raw/PanNuke/fold3/Fold 3/masks/fold3/masks.npy"
    dest_path = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/val"
    prefix = "val"
    pn2inst_type(data_path, dest_path, prefix)


def convert2coco():
    dataset_name = "pannuke"
    data_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/train"
    save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/train/train_annotations.json"
    test_mode = False

    convert_to_coco(dataset_name, data_dir, save_path, test_mode=test_mode)

    dataset_name = "pannuke"
    data_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test"
    save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/test_annotations.json"
    # save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/annotations/test_annotations.json"
    test_mode = False

    convert_to_coco(dataset_name, data_dir, save_path, test_mode=test_mode)

    # dataset_name = "pannuke"
    # data_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/val"
    # save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/val/val_annotations.json"
    # # save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/annotations/test_annotations.json"
    # test_mode = True

    # convert_to_coco(dataset_name, data_dir, save_path, test_mode=test_mode)


def gen_dist_map():
    data_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/train"
    pannuke = PanNuke(data_dir)
    insts_path = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/train/inst"
    save_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/train/dist_map"
    pannuke.gen_dist_map(insts_path, save_dir)


if __name__ == "__main__":
    # pannuke2mask()
    convert2coco()
    # gen_dist_map()
