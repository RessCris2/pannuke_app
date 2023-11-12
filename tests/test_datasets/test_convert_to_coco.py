import os
import sys

import pytest

sys.path.append("/root/autodl-tmp/pannuke_app/")
from src.data_process.convert2coco import convert_to_coco
from src.data_process.monusac2mask import convert


def consep_convert():
    dataset_name = "consep"
    data_dir = "/root/autodl-tmp/pannuke_app/datasets/raw/CoNSeP/CoNSeP/Train"
    save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/annotations/train_annotations.json"
    test_mode = True

    convert_to_coco(dataset_name, data_dir, save_path, test_mode=test_mode)

    dataset_name = "consep"
    data_dir = "/root/autodl-tmp/pannuke_app/datasets/raw/CoNSeP/CoNSeP/Test"
    save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/annotations/test_annotations.json"
    test_mode = True

    convert_to_coco(dataset_name, data_dir, save_path, test_mode=test_mode)


def monusac2mask():
    """convert monusac to seg_mask(.png) and inst_mask(.npy)
    and convert imgs to png format
    """

    data_path = "/root/autodl-tmp/pannuke_app/datasets/raw/MoNuSAC/train"
    dest_path = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/train"
    convert(data_path, dest_path)


if __name__ == "__main__":
    monusac2mask()

    # dataset_name = "monusac"
    # data_dir = "/root/autodl-tmp/pannuke_app/datasets/raw/CoNSeP/CoNSeP/Train"
    # save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/annotations/train_annotations.json"
    # test_mode = True

    # convert_to_coco(dataset_name, data_dir, save_path, test_mode=test_mode)

    # dataset_name = "monusac"
    # data_dir = "/root/autodl-tmp/pannuke_app/datasets/raw/CoNSeP/CoNSeP/Test"
    # save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/annotations/test_annotations.json"
    # test_mode = True

    # convert_to_coco(dataset_name, data_dir, save_path, test_mode=test_mode)
