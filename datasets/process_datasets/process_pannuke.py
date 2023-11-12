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
    data_path = "/root/autodl-tmp/pannuke_app/datasets/raw/PanNuke/test"
    dest_path = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test"
    pn2img(data_path, dest_path)
    pn2inst_type(data_path, dest_path)


def convert2coco():
    dataset_name = "PanNuke"
    data_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/train"
    save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/train/train_annotations.json"
    test_mode = True

    convert_to_coco(dataset_name, data_dir, save_path, test_mode=test_mode)

    dataset_name = "PanNuke"
    data_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test"
    save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/test_annotations.json"
    # save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/annotations/test_annotations.json"
    test_mode = True

    convert_to_coco(dataset_name, data_dir, save_path, test_mode=test_mode)


def gen_dist_map():
    data_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/train"
    PanNuke = PanNuke(data_dir)
    insts_path = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/train/inst"
    save_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/train/dist_map"
    PanNuke.gen_dist_map(self, insts_path, save_dir)


if __name__ == "__main__":
    # PanNuke2mask()
    convert2coco()
    gen_dist_map()
