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
from dataset import get_dataset
from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = True

    # win_size = [540, 540] # Monusac has more diverse image sizes, set 270 for now. then image with size smaller than 270 will be passed.
    # but this means hovernet then will have no augmented patch. 350 may be better choice.
    win_size = [350, 350]
    step_size = [164, 164]
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_name = "monusac"
    save_root = "dataset/training_data/%s/" % dataset_name

    # a dictionary to specify where the dataset path should be
    dataset_info = {
        "train": {
            "img": (
                ".png",
                "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/train/imgs",
            ),
            "ann": (
                ".png",
                "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/train/imgs",
            ),
        },
        "valid": {
            "img": (
                ".png",
                "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/test/imgs",
            ),
            "ann": (
                ".png",
                "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/test/imgs",
            ),
        },
    }

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        ann_ext, ann_dir = split_desc["ann"]

        out_dir = "%s/%s/%s/%dx%d_%dx%d/" % (
            save_root,
            dataset_name,
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
        )
        file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
        file_list.sort()  # ensure same ordering across platform

        rm_n_mkdir(out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem

            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            if img.shape[0] < win_size[0] or img.shape[1] < win_size[1]:
                continue
            ann = parser.load_ann(
                "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
            )

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
