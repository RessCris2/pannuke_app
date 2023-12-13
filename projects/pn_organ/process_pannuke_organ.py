""" 将原始数据集按 organ 划分为train, test
"""
import os
import random
import shutil
import sys
from collections import Counter
from os import path as osp

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append("/root/autodl-tmp/pannuke_app/")
from src.data_process.pannuke2mask import pn2img, pn2inst_type


def split_index(save_dir, types, train_types):
    """将原始数据集按 organ 划分为train, test, 选择 train_types, test_types 相应的index 作为训练集和测试集的index"""
    train_set = []
    test_set = []
    for i, t in enumerate(types):
        if t in train_types:
            train_set.append(i)
        else:
            test_set.append(i)
    train_set = np.array(train_set)
    test_set = np.array(test_set)
    print(len(train_set), len(test_set))
    np.save(f"{save_dir}/train_set.npy", train_set)
    np.save(f"{save_dir}/test_set.npy", test_set)


type_names = np.array(
    [
        "Breast",
        "Colon",
        "Lung",
        "Kidney",
        "Prostate",
        "Bladder",
        "Stomach",
        "Ovarian",
        "Esophagus",
        "Pancreatic",
        "Uterus",
        "Thyroid",
        "Skin",
        "Cervix",
        "Adrenal_gland",
        "Bile-duct",
        "Testis",
        "HeadNeck",
        "Liver",
    ]
)

"""
第0次split ['Adrenal_gland' 'Bile-duct' 'Breast' 'Cervix' 'Esophagus' 'HeadNeck'
 'Kidney' 'Liver' 'Lung' 'Skin' 'Stomach' 'Testis' 'Thyroid'] ['Colon' 'Prostate' 'Bladder' 'Ovarian' 'Pancreatic' 'Uterus']
1931 725
1822 701
1853 869
第1次split ['Bile-duct' 'Bladder' 'Cervix' 'Colon' 'Esophagus' 'HeadNeck' 'Kidney'
 'Lung' 'Ovarian' 'Pancreatic' 'Prostate' 'Testis' 'Thyroid'] ['Breast' 'Stomach' 'Uterus' 'Skin' 'Adrenal_gland' 'Liver']
1504 1152
1419 1104
1447 1275
第2次split ['Adrenal_gland' 'Bile-duct' 'Bladder' 'Breast' 'Colon' 'Liver' 'Ovarian'
 'Pancreatic' 'Prostate' 'Skin' 'Stomach' 'Testis' 'Uterus'] ['Lung' 'Kidney' 'Esophagus' 'Thyroid' 'Cervix' 'HeadNeck']
2079 577
1981 542
2196 526
"""


# 1. 划分数据集 v1
# 19
# 循环三次 split
def fetch_split_index(type_names=type_names):
    for split in range(3):
        n = len(type_names)
        # train_index = np.random.choice(list(range(n)), 15, replace=False)
        if split == 0:
            test_index = [1, 4, 5, 7, 9, 10]
        elif split == 1:
            test_index = [0, 6, 10, 12, 14, 18]
        else:
            test_index = [2, 3, 8, 11, 13, 17]
        test_types = type_names[test_index]
        train_types = np.setdiff1d(type_names, test_types)
        print(f"第{split}次split", train_types, test_types)

        root_dir = "/root/autodl-tmp/pannuke_app/projects/pn_organ"

        # 循环3个数据集
        for i in range(3):
            save_dir = f"{root_dir}/split_v{split+1}/fold{i+1}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            types = np.load(
                f"/root/autodl-tmp/pannuke_app/datasets/raw/PanNuke/Fold {i+1}/images/fold{i+1}/types.npy"
            )

            split_index(save_dir, types, train_types)


def process_split_data():
    # 将相应的 train,test split index 提取为 img, seg_mask, inst
    save_root0 = "/root/autodl-tmp/pannuke_app/projects/pn_organ"
    data_root = "/root/autodl-tmp/pannuke_app/datasets/raw/PanNuke"

    # 三次循环，每次处理一个 split
    for split_index in range(3):
        # 从 三个fold 中分别读取数据
        save_root = f"{save_root0}/split_v{split_index+1}"
        for fold_index in range(3):
            index_dir = f"{save_root}/fold{fold_index+1}"
            train_index = np.load(f"{index_dir}/train_set.npy")
            test_index = np.load(f"{index_dir}/test_set.npy")

            # 读取数据
            # 1. img
            img_data = np.load(
                f"{data_root}/Fold {fold_index+1}/images/fold{fold_index+1}/images.npy"
            )

            # 2. mask
            mask_data = np.load(
                f"{data_root}/Fold {fold_index+1}/masks/fold{fold_index+1}/masks.npy"
            )

            save_dir = f"{save_root}/split_v{split_index+1}/training_data/train"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            images = img_data[train_index]
            masks = mask_data[train_index]
            # mask 的处理比较麻烦
            pn2img(images, save_dir, f"train_fold{fold_index+1}")
            pn2inst_type(masks, save_dir, f"train_fold{fold_index+1}")

            save_dir = f"{save_root}/split_v{split_index+1}/training_data/test"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            images = img_data[test_index]
            masks = mask_data[test_index]
            # mask 的处理比较麻烦
            pn2img(images, save_dir, f"test_fold{fold_index+1}")
            pn2inst_type(masks, save_dir, f"test_fold{fold_index+1}")


def convert_to_coco():
    dataset_name = "pannuke"
    for split_index in range(3):
        data_root = f"/root/autodl-tmp/pannuke_app/projects/pn_organ/split_v{split_index+1}/training_data"
        data_dir = f"{data_root}/train"
        save_path = f"{data_root}/train/train_annotations.json"
        test_mode = False
        convert_to_coco(dataset_name, data_dir, save_path, test_mode=test_mode)

        data_dir = f"{data_root}/test"
        save_path = f"{data_root}/test/test_annotations.json"
        test_mode = False
        convert_to_coco(dataset_name, data_dir, save_path, test_mode=test_mode)


if __name__ == "__main__":
    # fetch_split_index(type_names=type_names)
    process_split_data()
    # convert_to_coco()
