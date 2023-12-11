"""
读取 patch 后的数据
"""

"""处理模型输入为 maskrcnn 的输入要求
"""

import glob
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


def remove_empty_img(data_dir):
    imgs = glob.glob(f'{data_dir}/*.npy')
    for img_path in imgs:
        data = np.load(img_path)
        inst_map = data[..., 3]
        if np.sum(inst_map) == 0:
            os.remove(img_path)
            print(f"remove empty {img_path}")
        

        obj_ids = np.unique(inst_map)
    
        masks = []
        obj_ids = obj_ids[1:]
        # get type labels
        for obj_id in obj_ids:
            mask = np.where(inst_map== obj_id,1,0)
             # remove small masks
            if np.sum(mask) < 16:
                continue
            masks.append(mask)
            
        if len(masks) == 0:
            print(f"remove too small object {img_path}")
            os.remove(img_path)
        

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    # if train:
        # transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class CoNSePDataset(torch.utils.data.Dataset):
    # def __init__(self, imgs, masks, transforms):
    def __init__(self, data_dir, transforms=None):
        # 会在这里进行一部分的数据增强？
        self.transforms = transforms 
       
        self.imgs = glob.glob(f'{data_dir}/*.npy') #sorted(glob.glob('/kaggle/input/hubmap-making-dataset/train/image/*.png'))
        # self.masks = glob.glob(f'{masks_dir}/*.png') #sorted(glob.glob('/kaggle/input/hubmap-making-dataset/train/mask/*.png'))

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        data = np.load(img_path)
        img_array = data[..., :3]
        H, W = img_array.shape[:2]
        img = Image.fromarray(img_array.astype(np.uint8))
        inst_map = data[..., 3]
        type_mask = data[..., 4]

        obj_ids = np.unique(inst_map)
        assert len(obj_ids) >1, "obj_ids is empty"
    
        masks = []
        labels = []
        boxes = []
        obj_ids = obj_ids[1:]
        # get type labels
        for obj_id in obj_ids:
            mask = np.where(inst_map== obj_id,1,0)
             # remove small masks
            if np.sum(mask) < 16:
                continue

            class_ids = np.unique(np.where(mask==1, type_mask, 0))
            if len(class_ids) > 1:
                class_id = class_ids[1]
            else:
                continue

            # get bounding box coordinates for each mask, use 2 more pixels to avoid same y,x
            pos = np.nonzero(mask)
            xmin = np.clip( np.min(pos[1])-1, 0, W)
            xmax = np.clip(np.max(pos[1])+1, 0, W) 
            ymin = np.clip(np.min(pos[0])-1,0, H)
            ymax = np.clip(np.max(pos[0])+1, 0, H)

            
            
            boxes.append([xmin, ymin, xmax, ymax])

           
            masks.append(mask)
            labels.append(class_id)
        masks = np.array(masks)
        labels = np.array(labels)
        boxes = np.array(boxes)
        # print(len(masks), len(labels))

        if len(boxes)==0:
            print('xxx')
        assert len(boxes) > 0, 'boxes is empty'
        
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        # labels = torch.ones((num_objs,), dtype=torch.int64)

        masks = torch.as_tensor(masks, dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # image_id = torch.tensor([idx])
        image_id = idx
        try:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            #print(area,area.shape,area.dtype)
        except:
            area = torch.tensor([[0],[0]])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(masks),), dtype=torch.int64)
        
        #print(masks.shape)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            # img, target = self.transforms(img, target)
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    data_dir = "/root/autodl-tmp/pannuke_app/projects/consep/maskrcnn_v23/training_data/valid/256x256_80x80"
    # data_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/train/"
    transform = get_transform(train=True)
    dataset = CoNSePDataset(data_dir, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    for i, (img, target) in enumerate(dataloader):
        print(img.shape)
        # print(target)
        # break
    # data_dir = "/root/autodl-tmp/pannuke_app/projects/consep/maskrcnn_v23/training_data/valid/256x256_80x80"
    # remove_empty_img(data_dir)
