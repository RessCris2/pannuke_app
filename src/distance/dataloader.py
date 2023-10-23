"""处理数据输入为 pannuke 
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import glob
from pathlib import Path
from scipy.ndimage import distance_transform_cdt
import sys
# sys.path.append("/root/autodl-tmp/archive/v2/metrics")
from src.core.utils import load_img

"""
  如果只是 UNet 的输入的话，应该只需要输入 images, mask, 每个channel上是分类吗?
  我个人认为应该是处理为 同样 channel 表达分类的label会比较好。
"""
def get_id_list(file_dir, postfix='jpg'):
    file_path = os.path.join(file_dir, "*.{}".format(postfix))
    files = glob.glob(file_path)
    
    assert len(files) > 1
    ids = [Path(file).stem for file in files]
    return ids


def distancewithoutnormalise(bin_image):
    res = np.zeros_like(bin_image)
    for j in range(1, bin_image.max() + 1):
        one_cell = np.zeros_like(bin_image)
        one_cell[bin_image == j] = 1
        one_cell = distance_transform_cdt(one_cell)
        res[bin_image == j] = one_cell[bin_image == j]
    res = res.astype('uint8')
    return res

class PanNukeDataset(Dataset):
    CLASS = ['Background','Neoplastic','Inflammatory','Connective','Dead','Epithelial']
    def __init__(self, images_dir, masks_dir, classes=CLASS, augmentation=None, preprocessing=None,):

        # Obtain the number of files with the specified suffix 'jpg'
        self.ids = get_id_list(images_dir, 'jpg')
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.classes = classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        base = self.ids[index]
        img = cv2.imread( os.path.join(self.images_dir, "{}.jpg".format(base)))
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread( os.path.join(self.masks_dir, "{}.png".format(base)), 0)
        label = np.stack([np.where(mask==i, 1, 0 ) for i in range(len( self.classes ))])  # label CHW
        return (torch.tensor(image), torch.tensor(label))

def data_aug():
    trans_fn = transforms.Compose([transforms.ToTensor()])
    return trans_fn

class DISTDataset(Dataset):
    def __init__(self, images_dir, masks_dir, classes, augmentation=data_aug, preprocessing=None,):
        self.ids = get_id_list(images_dir, 'jpg')
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.classes = classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        base = self.ids[index]
        img_path = os.path.join(self.images_dir, "{}.jpg".format(base))
        img = load_img( img_path )
        if self.augmentation is not None:
            img = self.augmentation()(img)
        mask = cv2.imread( os.path.join(self.masks_dir, "{}.png".format(base)), 0)
        ## The label in DIST needs to be converted to a distance map
        label = distancewithoutnormalise(mask)  
        return img, torch.tensor(label, dtype=torch.float32), img_path