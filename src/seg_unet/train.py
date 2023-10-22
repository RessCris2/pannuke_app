# -*- coding:utf-8 -*-

import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import glob

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU


## 预处理数据集 
class PanNukeDataset(BaseDataset):
    """PanNuke Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        # images_dir (str): path to images folder
        # masks_dir (str): path to segmentation masks folder
        # class_values (list): values of classes to extract from segmentation mask
        # augmentation (albumentations.Compose): data transfromation pipeline 
        #     (e.g. flip, scale, etc.)
        # preprocessing (albumentations.Compose): data preprocessing 
        #     (e.g. noralization, shape manipulation, etc.)
    
    """
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            num_classes,
#             classes= ['Background','Neoplastic','Inflammatory','Connective','Dead Cells','Epithelial'], 
        #     augmentation=None, 
        #     preprocessing=None,
    ):
        img_path = os.path.join(images_dir, "*.jpg")
        img_list = glob.glob(img_path)
        # self.ids = os.listdir(images_dir)
        self.images_fps = img_list

        mask_path = os.path.join(masks_dir, "*.png")
        mask_list = glob.glob(mask_path)
        self.masks_fps = mask_list
        self.num_classes = num_classes
        assert len(self.images_fps) > 0
        assert len(self.masks_fps) > 0
        assert len(self.images_fps) == len(self.masks_fps)
        # 一次性读取是不是会消耗太多内存？后续可以考虑改为文件夹内分批读取？
        # self.images =  np.load(images_dir)
        # self.masks = np.load(masks_dir) # 确认一下 masks 是否都是小于6
        
        # convert str names to class values on masks
#         self.class_values = [self.CLASSES.index(cls) for cls in self.CLASSES]
        self.class_values = list(range(num_classes))
        
#         self.augmentation = augmentation
#         self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        # mask = np.load(self.masks_fps[i])
        mask = cv2.imread(self.masks_fps[i],0)
        mask = np.where(mask >= self.num_classes, 0, mask)
        assert mask.max() < self.num_classes
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        mask = np.transpose(mask, (2, 0, 1))
        # image = self.images[i,...]
        # image = np.transpose(image, (2, 0, 1))
        # mask = self.masks[i,...,[5, 0, 1, 2, 3, 4]]
        # mask = np.where(mask>0, 1, 0)
        
        
        # apply augmentations
#         if self.augmentation:
#             sample = self.augmentation(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
        
#         # apply preprocessing
#         if self.preprocessing:
#             sample = self.preprocessing(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
            
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)
        
    def __len__(self):
        return len(self.images_fps)


def train_model(num_classes, model_name, dataset_name):
    epochs = 40
    
    train_dataset = PanNukeDataset(
                        images_dir = '/root/autodl-tmp/archive/datasets/{}/patched/coco_format/images/train'.format(dataset_name), 
                        masks_dir = '/root/autodl-tmp/archive/datasets/{}/patched/coco_format/seg_mask/train/'.format(dataset_name), 
                        num_classes = num_classes
                        # augmentation=get_training_augmentation(), 
                        # preprocessing=get_preprocessing(preprocessing_fn),
                        # classes=CLASSES,
                        )

    valid_dataset = PanNukeDataset(
                        images_dir= '/root/autodl-tmp/archive/datasets/pannuke/patched/coco_format/images/test',
                        masks_dir = '/root/autodl-tmp/archive/datasets/pannuke/patched/coco_format/seg_mask/test', 
                        num_classes=num_classes
                        # augmentation=get_validation_augmentation(), 
                        # preprocessing=get_preprocessing(preprocessing_fn),
                        # classes=CLASSES,
                    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = smp.Unet(
                encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=num_classes,            # model output channels (number of classes in your dataset)
                activation='softmax'
                )

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    loss = DiceLoss()
    metrics = [
        IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
    )

    # train model for 40 epochs
    max_score = 0

    for i in range(0, epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, '/root/autodl-tmp/archive/v2/model_data/{}/{}/model_{}.pth'.format(model_name, dataset_name, i))
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')


def train_wb(config, num_classes, model_name, dataset_name, train_dir, test_dir, log_dir):
    epochs = config.epochs
    
    train_dataset = PanNukeDataset(
                        images_dir = f'{train_dir}/images/train', 
                        masks_dir = f'{train_dir}/seg_mask/train/', 
                        num_classes = num_classes
                        # augmentation=get_training_augmentation(), 
                        # preprocessing=get_preprocessing(preprocessing_fn),
                        # classes=CLASSES,
                        )

    valid_dataset = PanNukeDataset(
                        images_dir= f'{test_dir}/images/test',
                        masks_dir = f'{test_dir}/seg_mask/test', 
                        num_classes=num_classes
                        # augmentation=get_validation_augmentation(), 
                        # preprocessing=get_preprocessing(preprocessing_fn),
                        # classes=CLASSES,
                    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = smp.Unet(
                encoder_name=config.encoder_name,        # choose encoder, e.g. resnet34, mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=num_classes,            # model output channels (number of classes in your dataset)
                activation='softmax'
                )

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    loss = DiceLoss()
    metrics = [
        IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=config.lr),
    ])

    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
    )

    # train model for 40 epochs
    max_score = 0

    for i in range(0, epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, f'{log_dir}/model_{i}.pth')
            print('Model saved!')
            
        # if i == 25:
        #     optimizer.param_groups[0]['lr'] = 1e-5
        #     print('Decrease decoder learning rate to 1e-5!')
if __name__ == "__main__":
    # train_model(num_classes=6)
    train_model(num_classes=5, model_name='seg_unet', dataset_name='monusac')
    train_model(num_classes=5, model_name='seg_unet', dataset_name='consep')
    # train_model(num_classes=6, model_name='seg_unet', dataset_name='pannuke')
    


    