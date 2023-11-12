import os
import sys

import pytest

sys.path.append("/root/autodl-tmp/pannuke_app/")
from src.data_process.convert2coco import convert_to_coco
from src.data_process.data_transformer import CoNSeP


def convert2coco():
    dataset_name = "consep"
    data_dir = "/root/autodl-tmp/pannuke_app/datasets/raw/CoNSeP/CoNSeP/Train"
    # save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/annotations/train_annotations.json"
    save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/train/train_annotations.json"
    test_mode = False

    convert_to_coco(dataset_name, data_dir, save_path, test_mode=test_mode)

    dataset_name = "consep"
    data_dir = "/root/autodl-tmp/pannuke_app/datasets/raw/CoNSeP/CoNSeP/Test"
    save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/test_annotations.json"
    test_mode = False

    convert_to_coco(dataset_name, data_dir, save_path, test_mode=test_mode)


def gen_mask():
    data_dir = "/root/autodl-tmp/pannuke_app/datasets/raw/CoNSeP/CoNSeP/Train"
    consep = CoNSeP(data_dir)
    labels_path = "/root/autodl-tmp/pannuke_app/datasets/raw/CoNSeP/CoNSeP/Train/Labels"
    save_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/train/seg_mask"
    consep.gen_mask(labels_path, save_dir)

    data_dir = "/root/autodl-tmp/pannuke_app/datasets/raw/CoNSeP/CoNSeP/Test"
    consep = CoNSeP(data_dir)
    labels_path = "/root/autodl-tmp/pannuke_app/datasets/raw/CoNSeP/CoNSeP/Test/Labels"
    save_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/seg_mask"
    consep.gen_mask(labels_path, save_dir)


def gen_dist_map():
    data_dir = "/root/autodl-tmp/pannuke_app/datasets/raw/CoNSeP/CoNSeP/Train"
    consep = CoNSeP(data_dir)
    labels_path = "/root/autodl-tmp/pannuke_app/datasets/raw/CoNSeP/CoNSeP/Train/Labels"
    save_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/train/dist_map"
    consep.gen_dist_map(labels_path, save_dir)

    data_dir = "/root/autodl-tmp/pannuke_app/datasets/raw/CoNSeP/CoNSeP/Test"
    consep = CoNSeP(data_dir)
    labels_path = "/root/autodl-tmp/pannuke_app/datasets/raw/CoNSeP/CoNSeP/Test/Labels"
    save_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/dist_map"
    consep.gen_dist_map(labels_path, save_dir)


if __name__ == "__main__":
    convert2coco()
    gen_mask()
    gen_dist_map()
