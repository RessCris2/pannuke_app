"""
extract patch
convert to coco
"""
import sys

sys.path.append("/root/autodl-tmp/pannuke_app/")
from src.data_process.extract_fun import extract_patches


def patch_monusac():
    dataset_name = 'monusac'
    # raw_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/train/"
    # save_root = "/root/autodl-tmp/pannuke_app/projects/monusac/training_data/train/"
    # extract_patches(dataset_name, raw_dir, save_root)
    # print("train patch done")

    raw_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/test/"
    save_root = "/root/autodl-tmp/pannuke_app/projects/monusac/training_data/test/"
    extract_patches(dataset_name, raw_dir, save_root)
    print("test patch done")


def convert_patch_to_coco():
    dataset_name = "monusac"
    root_dir = "/root/autodl-tmp/pannuke_app/projects/monusac/training_data/"
    data_dir = f"{root_dir}/train/"
    save_path = f"{root_dir}/train/train_annotations.json"
    test_mode = False

    convert_to_coco(dataset_name, data_dir, save_path, test_mode=test_mode)

    dataset_name = f"{root_dir}/test"
    save_path = f"{root_dir}/test/test_annotations.json"
    # save_path = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/annotations/test_annotations.json"
    test_mode = False
    convert_to_coco(dataset_name, data_dir, save_path, test_mode=test_mode)


if __name__ == "__main__":
    patch_monusac()
    # convert_patch_to_coco()