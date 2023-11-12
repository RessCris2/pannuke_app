"""temp file to generate segmask for each image in the dataset consep
"""
import glob
import os

import cv2
import scipy.io as sio


# img = cv2.imread("/home/pannuke_app/train/datasets/CoNSeP/Test/seg_mask/test_4.png", flags=0)
# 要通过设定 flags 才能正确读取。这里后面校验mmseg中的处理
def generate_segmask(labels_path):
    """generate segmask for each image in the dataset consep"""

    save_dir = labels_path.replace("Labels", "seg_mask")
    os.makedirs(save_dir, exist_ok=True)
    paths = glob.glob(f"{labels_path}/*.mat")
    for label_path in paths:
        ann_type = sio.loadmat(label_path)["type_map"]

        # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
        # If own dataset is used, then the below may need to be modified
        ann_type[(ann_type == 3) | (ann_type == 4)] = 3
        ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4
        type_path = label_path.replace("Labels", "seg_mask").replace("mat", "png")
        cv2.imwrite(type_path, ann_type, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        # cv2.imwrite(type_path, type_map) # tha same as above


if __name__ == "__main__":
    if os.path.exists("/home/pannuke_app/"):
        labels_path = "/home/pannuke_app/train/datasets/CoNSeP/Test/Labels"
    else:
        labels_path = "/root/autodl-tmp/pannuke_app/train/datasets/CoNSeP/Test/Labels"
    generate_segmask(labels_path)
