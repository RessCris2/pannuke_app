""" 将原始数据集按 organ 划分为train, test
"""
import os
import shutil
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter


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

# 1. 划分数据集 v1
# 19
# 循环三次 split
for split in range(3):
    n = len(type_names)
    train_index = np.random.choice(list(range(n)), 15, replace=False)
    train_types = type_names[train_index]
    test_types = np.setdiff1d(type_names, train_types)
    print(f"第{split}次split", train_types, test_types)

    root_dir = "/home/pannuke_app/projects/pannuke_organ"

    # 循环3个数据集
    for i in range(3):
        save_dir = f"{root_dir}/split_v{split}/fold{i+1}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        types = np.load(
            f"/home/pannuke_app/datasets/raw/PanNuke/Fold {i+1}/images/fold{i+1}/types.npy"
        )

        split_index(save_dir, types, train_types, test_types)
