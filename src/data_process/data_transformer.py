"""As for CoNSeP dataset, first has to make the format be [img, inst_id, type_id], 
with shape [N, h, w, 5], and then extract to generate patches for training and testing.
"""
import glob
import os
import pathlib

import cv2
import numpy as np
import scipy.io as sio
from scipy.ndimage import distance_transform_cdt

from ..utils import find_files


def distancewithoutnormalise(bin_image):
    res = np.zeros_like(bin_image)
    for j in range(1, bin_image.max() + 1):
        one_cell = np.zeros_like(bin_image)
        one_cell[bin_image == j] = 1
        one_cell = distance_transform_cdt(one_cell)
        res[bin_image == j] = one_cell[bin_image == j]
    res = res.astype("uint8")
    return res


class __AbstractDataset(object):
    """Abstract class for interface of subsequent classes.
    Main idea is to encapsulate how each dataset should parse
    their images and annotations.

    """

    def load_img(self, path):
        raise NotImplementedError

    def load_ann(self, path, with_type=False):
        raise NotImplementedError

    def load_category(self):
        raise NotImplementedError


class CoNSeP(__AbstractDataset):
    """Defines the CoNSeP dataset as originally introduced in:

    Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak,
    and Nasir Rajpoot. "Hover-Net: Simultaneous segmentation and classification of nuclei in
    multi-tissue histology images." Medical Image Analysis 58 (2019): 101563

    """

    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        # TODO
        self.imgs = glob.glob(f"{data_dir}/Images/*.png")
        # self.imgs = find_files(data_dir, ext="png")
        self.labels = find_files(data_dir, ext="mat")

        # init category attribute
        self.load_category()

    def load_ann_path(self, img_path):
        """get correspondign label path for the given img_path"""
        return img_path.replace("Images", "Labels").replace("png", "mat")

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, img_path, with_type=True):
        path = self.load_ann_path(img_path)
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)["inst_map"]
        if with_type:
            ann_type = sio.loadmat(path)["type_map"]

            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            ann_type[(ann_type == 3) | (ann_type == 4)] = 3
            ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype("int32")
            return ann
            # return ann_inst, ann_type
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype("int32")

        return ann

    def gen_inst_map(self, labels_path, save_dir):
        """generate segmask for each image in the dataset consep
        for the input of MMSegmentation
        Params:
        """
        # save_dir = labels_path.replace("Labels", "seg_mask")
        os.makedirs(save_dir, exist_ok=True)
        paths = glob.glob(f"{labels_path}/*.mat")
        for label_path in paths:
            basename = pathlib.Path(label_path).stem
            ann_inst = sio.loadmat(label_path)["inst_map"]
            ann_inst = ann_inst.astype("int32")
            inst_path = f"{save_dir}/{basename}.npy"
            np.save(inst_path, ann_inst)

    def gen_segmask(self, labels_path, save_dir):
        """generate segmask for each image in the dataset consep
        for the input of MMSegmentation
        Params:

        """
        # save_dir = labels_path.replace("Labels", "seg_mask")
        os.makedirs(save_dir, exist_ok=True)
        paths = glob.glob(f"{labels_path}/*.mat")
        for label_path in paths:
            basename = pathlib.Path(label_path).stem
            ann_type = sio.loadmat(label_path)["type_map"]

            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            ann_type[(ann_type == 3) | (ann_type == 4)] = 3
            ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4
            # type_path = label_path.replace("Labels", "seg_mask").replace("mat", "png")
            type_path = f"{save_dir}/{basename}.png"
            cv2.imwrite(type_path, ann_type, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def gen_dist_map(self, labels_path, save_dir):
        """generate distance transformed map for each image in the dataset consep
        for the input of MMSegmentation
        every pixel value means the nearest distance to the boundary of the instance
        """

        # save_dir = labels_path.replace("Labels", "seg_mask")
        os.makedirs(save_dir, exist_ok=True)
        paths = glob.glob(f"{labels_path}/*.mat")
        for label_path in paths:
            basename = pathlib.Path(label_path).stem
            # if need dist transform, we need to use inst map for transformation
            ann_inst = sio.loadmat(label_path)["inst_map"]
            ann_inst = ann_inst.astype("uint8")
            ann_dist = distancewithoutnormalise(ann_inst)
            dist_path = f"{save_dir}/{basename}.png"
            cv2.imwrite(dist_path, ann_dist, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            # cv2.imwrite(type_path, type_map) # tha same as above

    def load_category(self):
        """Customize the category information of the data set into coco
        category format
        """
        self.category = [
            {
                "id": 1,
                "name": "Inflammatory",
                "supercategory": "CoNSeP",
            },
            {
                "id": 2,
                "name": "Healthy_epithelial",
                "supercategory": "CoNSeP",
            },
            {
                "id": 3,
                "name": "Epithelial",
                "supercategory": "CoNSeP",
            },
            {
                "id": 4,
                "name": "Spindle-shaped",
                "supercategory": "CoNSeP",
            },
        ]


class PanNuke(__AbstractDataset):
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        # TODO
        self.imgs = glob.glob(f"{data_dir}/imgs/*.png")
        # self.imgs = find_files(data_dir, ext="png")
        # self.labels = find_files(data_dir, ext="mat")

        # init category attribute
        self.load_category()

    def load_ann_path(self, img_path):
        """get correspondign label path for the given img_path"""
        inst_path = img_path.replace("imgs", "inst").replace("png", "npy")
        seg_mask_path = img_path.replace("imgs", "seg_mask")  # .replace("png", "npy")
        return inst_path, seg_mask_path

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, img_path, with_type=True):
        """
        输入的是 img_path
        """
        # assumes that ann is HxW
        # ann_inst = sio.loadmat(path)["inst_map"]
        inst_path, seg_mask_path = self.load_ann_path(img_path)
        ann_inst = np.load(inst_path)

        if with_type:
            # ann_type = np.load(seg_mask_path)
            ann_type = cv2.imread(seg_mask_path, 0)
            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype("int32")
            return ann
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype("int32")

        return ann

    def gen_dist_map(self, insts_path, save_dir):
        """generate distance transformed map for each image in the dataset consep
        for the input of MMSegmentation
        every pixel value means the nearest distance to the boundary of the instance
        """

        # save_dir = labels_path.replace("Labels", "seg_mask")
        os.makedirs(save_dir, exist_ok=True)
        paths = glob.glob(f"{insts_path}/*.npy")
        for label_path in paths:
            basename = pathlib.Path(label_path).stem
            # if need dist transform, we need to use inst map for transformation
            ann_inst = np.load(label_path)
            ann_inst = ann_inst.astype("uint8")
            ann_dist = distancewithoutnormalise(ann_inst)
            dist_path = f"{save_dir}/{basename}.png"
            cv2.imwrite(dist_path, ann_dist, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def load_category(self):
        """Customize the category information of the data set into coco
        category format
        """
        self.category = [
            {
                "id": 1,
                "name": "Neoplastic",
                "supercategory": "PanNuke",
            },
            {
                "id": 2,
                "name": "Inflammatory",
                "supercategory": "PanNuke",
            },
            {
                "id": 3,
                "name": "Connective",
                "supercategory": "PanNuke",
            },
            {
                "id": 4,
                "name": "Dead",
                "supercategory": "PanNuke",
            },
            {
                "id": 5,
                "name": "Epithelial",
                "supercategory": "PanNuke",
            },
        ]


class MoNuSAC(__AbstractDataset):
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        # TODO
        self.imgs = glob.glob(f"{data_dir}/imgs/*.png")
        # self.imgs = find_files(data_dir, ext="png")
        # self.labels = find_files(data_dir, ext="mat")

        # init category attribute
        self.load_category()

    def load_ann_path(self, img_path):
        """get correspondign label path for the given img_path"""
        inst_path = img_path.replace("imgs", "inst").replace("png", "npy")
        seg_mask_path = img_path.replace("imgs", "seg_mask")  # .replace("png", "npy")
        return inst_path, seg_mask_path

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, img_path, with_type=True):
        """
        输入的是 img_path
        """
        # assumes that ann is HxW
        # ann_inst = sio.loadmat(path)["inst_map"]
        inst_path, seg_mask_path = self.load_ann_path(img_path)
        ann_inst = np.load(inst_path)

        if with_type:
            ann_type = cv2.imread(seg_mask_path, 0)
            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype("int32")
            return ann
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype("int32")

        return ann

    def gen_dist_map(self, insts_path, save_dir):
        """generate distance transformed map for each image in the dataset consep
        for the input of MMSegmentation
        every pixel value means the nearest distance to the boundary of the instance
        """

        # save_dir = labels_path.replace("Labels", "seg_mask")
        os.makedirs(save_dir, exist_ok=True)
        paths = glob.glob(f"{insts_path}/*.npy")
        for label_path in paths:
            basename = pathlib.Path(label_path).stem
            # if need dist transform, we need to use inst map for transformation
            ann_inst = np.load(label_path)
            ann_inst = ann_inst.astype("uint8")
            ann_dist = distancewithoutnormalise(ann_inst)
            dist_path = f"{save_dir}/{basename}.png"
            cv2.imwrite(dist_path, ann_dist, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def load_category(self):
        self.category = [
            {
                "id": 1,
                "name": "Epithelial",
                "supercategory": "MoNuSAC",
            },
            {
                "id": 2,
                "name": "Lymphocyte",
                "supercategory": "MoNuSAC",
            },
            {
                "id": 3,
                "name": "Neutrophil",
                "supercategory": "MoNuSAC",
            },
            {
                "id": 4,
                "name": "Macrophage",
                "supercategory": "MoNuSAC",
            },
        ]


def get_transformer(name):
    """Return a pre-defined dataset object associated with `name`."""
    name_dict = {
        # "kumar": lambda: __Kumar(),
        # "cpm17": lambda: __CPM17(),
        "monusac": lambda: MoNuSAC,
        "consep": lambda: CoNSeP,
        "pannuke": lambda: PanNuke,
    }
    if name.lower() in name_dict:
        return name_dict[name]()
    else:
        assert False, "Unknown dataset `%s`" % name


if __name__ == "__main__":
    data_dir = "/home/pannuke_pre/datasets/original/CoNSeP/Test"
    data = CoNSeP(data_dir)
