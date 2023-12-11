import glob
import os
import pathlib

import cv2
import numpy as np
import scipy.io as sio
from scipy.ndimage import distance_transform_cdt


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


####
class __Kumar(__AbstractDataset):
    """Defines the Kumar dataset as originally introduced in:

    Kumar, Neeraj, Ruchika Verma, Sanuj Sharma, Surabhi Bhargava, Abhishek Vahadane,
    and Amit Sethi. "A dataset and a technique for generalized nuclear segmentation for
    computational pathology." IEEE transactions on medical imaging 36, no. 7 (2017): 1550-1560.

    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        assert not with_type, "Not support"
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann


####
class __CPM17(__AbstractDataset):
    """Defines the CPM 2017 dataset as originally introduced in:

    Vu, Quoc Dang, Simon Graham, Tahsin Kurc, Minh Nguyen Nhat To, Muhammad Shaban,
    Talha Qaiser, Navid Alemi Koohbanani et al. "Methods for segmentation and classification
    of digital microscopy tissue images." Frontiers in bioengineering and biotechnology 7 (2019).

    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        assert not with_type, "Not support"
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann


####
class __CoNSeP(__AbstractDataset):
    """Defines the CoNSeP dataset as originally introduced in:

    Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak,
    and Nasir Rajpoot. "Hover-Net: Simultaneous segmentation and classification of nuclei in
    multi-tissue histology images." Medical Image Analysis 58 (2019): 101563

    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=True):
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
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype("int32")

        return ann


class PanNuke(__AbstractDataset):
    # def __init__(self, data_dir) -> None:
    #     self.data_dir = data_dir
    #     # TODO
    #     self.imgs = glob.glob(f"{data_dir}/imgs/*.png")
    #     # self.imgs = find_files(data_dir, ext="png")
    #     # self.labels = find_files(data_dir, ext="mat")

    #     # init category attribute
    #     self.load_category()

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
    # def __init__(self, data_dir) -> None:
    #     self.data_dir = data_dir
    #     # TODO
    #     self.imgs = glob.glob(f"{data_dir}/imgs/*.png")
    #     # self.imgs = find_files(data_dir, ext="png")
    #     # self.labels = find_files(data_dir, ext="mat")

    #     # init category attribute
    #     self.load_category()

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
            if np.max(ann_type) > 4:
                print("ann max > 4")
                ann_type = np.where(ann_type > 4, 0, ann_type)
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


####
def get_dataset(name):
    """Return a pre-defined dataset object associated with `name`."""
    name_dict = {
        "kumar": lambda: __Kumar(),
        "cpm17": lambda: __CPM17(),
        "consep": lambda: __CoNSeP(),
        "monusac": lambda: MoNuSAC(),
        "pannuke": lambda: PanNuke(),
    }
    if name.lower() in name_dict:
        return name_dict[name]()
    else:
        assert False, "Unknown dataset `%s`" % name
