"""extract_patches.py

Patch extraction script.
"""
import sys

sys.path.append("/root/autodl-tmp/pannuke_app/src/models/hover")
import glob
import os
import pathlib
import re

import numpy as np
import tqdm
# from dataset import get_dataset
from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

sys.path.append("/root/autodl-tmp/pannuke_app/")
from src.data_process.data_transformer import get_transformer


def extract_patches(dataset_name, raw_dir, save_root, win_size = [256, 256],
                     step_size = [80, 80], extract_type = "mirror", type_classification = True):
    """
    extract patches from raw images and annotations
    output: npys/*.npy [256, 256, 5]: [imgx3, inst_map, type_map]
    """
    parser = get_transformer(dataset_name)(raw_dir)
    xtractor = PatchExtractor(win_size, step_size)       
    out_dir = f"{save_root}/npys/"
    file_list = glob.glob(f"{raw_dir}/imgs/*.png" )
    file_list.sort()  # ensure same ordering across platform

    rm_n_mkdir(out_dir)

    pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
    pbarx = tqdm.tqdm(
        total=len(file_list), bar_format=pbar_format, ascii=True, position=0
    )

    for file_idx, file_path in enumerate(file_list):
        if file_idx < 52:
            continue
        
        base_name = pathlib.Path(file_path).stem
        img = parser.load_img_for_patch(file_path)
        ann = parser.load_ann_for_patch(file_path)
        # *
        img = np.concatenate([img, ann], axis=-1)
        sub_patches = xtractor.extract(img, extract_type)

        pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbar = tqdm.tqdm(
            total=len(sub_patches),
            leave=False,
            bar_format=pbar_format,
            ascii=True,
            position=1,
        )

        for idx, patch in enumerate(sub_patches):
            np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
            pbar.update()
        pbar.close()
        # *

        pbarx.update()
    pbarx.close()

if __name__ == "__main__":
    dataset_name = 'monusac'
    raw_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/train/"
    save_root = "/root/autodl-tmp/pannuke_app/projects/monusac/training_data/train/"
    extract_patches(dataset_name, raw_dir, save_root)
    print("done")