"""extract_patches.py

Patch extraction script.
"""
import sys

sys.path.append("/root/autodl-tmp/pannuke_app/src/models/hover")
import glob
import os
import pathlib
import re

import cv2
import numpy as np
import tqdm

# from dataset import get_dataset
from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

sys.path.append("/root/autodl-tmp/pannuke_app/")
from src.data_process.data_transformer import get_transformer


def remove_empty_img(data_dir):
    imgs = glob.glob(f"{data_dir}/*.npy")
    for img_path in imgs:
        data = np.load(img_path)
        inst_map = data[..., 3]
        if np.sum(inst_map) == 0:
            os.remove(img_path)
            print(f"remove empty {img_path}")
            os.remove(img_path.replace("npys", "inst"))
            os.remove(img_path.replace("npys", "seg_mask").replace(".npy", ".png"))

        obj_ids = np.unique(inst_map)

        masks = []
        obj_ids = obj_ids[1:]
        # get type labels
        for obj_id in obj_ids:
            mask = np.where(inst_map == obj_id, 1, 0)
            # remove small masks
            if np.sum(mask) < 16:
                continue
            masks.append(mask)

        if len(masks) == 0:
            print(f"remove too small object {img_path}")
            os.remove(img_path)
            os.remove(img_path.replace("npys", "inst"))
            os.remove(img_path.replace("npys", "seg_mask").replace(".npy", ".png"))


def extract_patches(
    dataset_name,
    raw_dir,
    save_root,
    win_size=[256, 256],
    step_size=[80, 80],
    extract_type="mirror",
    type_classification=True,
):
    """
    extract patches from raw images and annotations
    output: npys/*.npy [256, 256, 5]: [imgx3, inst_map, type_map]

        是否也顺便生成了inst_map和type_map? unet 要使用seg_mask
    """
    parser = get_transformer(dataset_name)(raw_dir)
    xtractor = PatchExtractor(win_size, step_size)
    out_dir = f"{save_root}/npys/"
    inst_dir = f"{save_root}/inst/"
    seg_mask_dir = f"{save_root}/seg_mask/"
    img_dir = f"{save_root}/imgs/"

    file_list = glob.glob(f"{raw_dir}/imgs/*.png")
    file_list.sort()  # ensure same ordering across platform

    rm_n_mkdir(out_dir)
    rm_n_mkdir(inst_dir)
    rm_n_mkdir(seg_mask_dir)
    rm_n_mkdir(img_dir)

    pbar_format = (
        "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
    )
    pbarx = tqdm.tqdm(
        total=len(file_list), bar_format=pbar_format, ascii=True, position=0
    )

    for file_idx, file_path in enumerate(file_list):
        # if file_idx == 1:
        #     continue
        base_name = pathlib.Path(file_path).stem
        img = parser.load_img_for_patch(file_path)
        ann = parser.load_ann_for_patch(file_path)
        # *
        img = np.concatenate([img, ann], axis=-1)
        if img.shape[0] < win_size[0] or img.shape[1] < win_size[1]:
            continue
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
            inst_map = patch[..., 3]
            if np.sum(inst_map) == 0:
                print(f"remove empty patch")
                continue

            obj_ids = np.unique(inst_map)

            masks = []
            obj_ids = obj_ids[1:]
            # get type labels
            for obj_id in obj_ids:
                mask = np.where(inst_map == obj_id, 1, 0)
                # remove small masks
                if np.sum(mask) < 16:
                    continue
                else:
                    masks.append(mask)
                    break

            if len(masks) == 0:
                print(f"remove too small object patch")
                continue

            npy_path = "{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx)
            inst_path = npy_path.replace("npys", "inst")
            type_path = npy_path.replace("npys", "seg_mask").replace(".npy", ".png")
            img_path = npy_path.replace("npys", "imgs").replace(".npy", ".png")

            np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
            np.save(inst_path, patch[:, :, 3])
            cv2.imwrite(type_path, patch[:, :, 4])
            cv2.imwrite(img_path, patch[:, :, :3])

            pbar.update()
        pbar.close()
        # *

        pbarx.update()
    pbarx.close()


if __name__ == "__main__":
    dataset_name = "monusac"
    raw_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/train/"
    save_root = "/root/autodl-tmp/pannuke_app/projects/monusac/training_data/train/"
    extract_patches(dataset_name, raw_dir, save_root)
    print("done")
