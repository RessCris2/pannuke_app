# -*- coding:utf-8 -*-

import os
import sys

import numpy as np

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(dir_path)

import glob

import albumentations as A
import cv2
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision
import wandb
from segmentation_models_pytorch.losses.dice import DiceLoss
from segmentation_models_pytorch.losses.focal import FocalLoss
from segmentation_models_pytorch.losses.jaccard import JaccardLoss
from segmentation_models_pytorch.losses.soft_ce import SoftCrossEntropyLoss
from segmentation_models_pytorch.utils import losses  # import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms as t

# import sys
# sys.path.append("/root/autodl-tmp/")
from src.core.early_stop import EarlyStopping
from src.core.evaluate import evalute_overall_im
from src.core.pre_proc import aug_fn, process_fn
from src.core.utils import rm_n_mkdir


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
            augmentation=aug_fn(), 
            preprocessing=process_fn()
    ):
        img_path = os.path.join(images_dir, "*.jpg")
        img_list = glob.glob(img_path)
        self.images_fps = img_list

        mask_path = os.path.join(masks_dir, "*.png")
        mask_list = glob.glob(mask_path)
        self.masks_fps = mask_list
        self.num_classes = num_classes
        assert len(self.images_fps) > 0
        assert len(self.masks_fps) > 0
        assert len(self.images_fps) == len(self.masks_fps)
        self.class_values = list(range(num_classes))
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.transpose(image, (2, 0, 1))
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
        
        # as_tensor = t.ToTensor()
        # apply augmentations
        if self.augmentation:
            aug = self.augmentation(image=image, mask=mask)
            image, mask = aug['image'], torch.tensor(aug['mask'], dtype=torch.long)
        
#         # apply preprocessing
        if self.preprocessing:
            image = self.preprocessing(image)
            # image = self.preprocessing(image=image)
            
        # return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)
        return image, mask
        
    def __len__(self):
        return len(self.images_fps)


def train_wb(config, num_classes, model_name, dataset_name, dir_root, log_dir):
    epochs = config.epochs
    rm_n_mkdir(log_dir)
    train_dataset = PanNukeDataset(
                        images_dir = f'{dir_root}/images/train', 
                        masks_dir = f'{dir_root}/seg_mask/train/', 
                        num_classes = num_classes,
                        augmentation=aug_fn(), 
                        preprocessing=process_fn(),
                        )

    valid_dataset = PanNukeDataset(
                        images_dir= f'{dir_root}/images/test',
                        masks_dir = f'{dir_root}/seg_mask/test', 
                        num_classes=num_classes,
                        augmentation=None,          # test 的时候需要使用 augmentation 么？暂定不用
                        preprocessing=process_fn(),
                    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = smp.Unet(
                encoder_name=config.encoder_name,        # choose encoder, e.g. resnet34, mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=num_classes,            # model output channels (number of classes in your dataset)
                activation=config.activation
                )

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    if config.loss == "DiceLoss":
        # loss = DiceLoss(mode='multilabel')
        loss = losses.DiceLoss()
        key = 'dice_loss'
    elif config.loss == "JaccardLoss":
        # loss = JaccardLoss(mode='multilabel')
        loss = losses.JaccardLoss()
        key = 'jaccard_loss'
    # elif config.loss == "FocalLoss":
    #     loss = FocalLoss(mode='multilabel')
    #     key = 'focal_loss'
    # elif config.loss == "SoftCrossEntropyLoss":
    #     loss = SoftCrossEntropyLoss(mode='multilabel')
    #     key = 'softcrossentropyloss'
    else:
        print("loss setting is wrong!")

    # 对比最后的评估，这里有内置的评估
    metrics = [
        IoU(threshold=0.5),
    ]

    if config.optimizer == "adam":
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=config.lr)])
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD([dict(params=model.parameters(), lr=config.lr)])
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW([dict(params=model.parameters(), lr=config.lr)])
    else:
        print("optimizer setting is wrong!")

    # 初始化 early_stopping 对象
    patience = 7	# 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=True, delta=0.00001)	# 关于 EarlyStopping 的代码可先看博客后面的内容


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
        try:
            print('\nEpoch: {}'.format(i))
            model = model.to('cuda')
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            
            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(model, f'{log_dir}/model_{i}.pth')
                print('Model saved!')
            
            if i % config.val_interval == 0:
                dataset_name = dataset_name
                model_name = "seg_unet"
                pred_dir = f"/root/autodl-tmp/datasets/{dataset_name}/images/test"
                
                # columns = ['map_50', 'map_75'] + ['acc', 'f1'] + [str(i) for i in type_uid_list] + ['dice','aji', 'aji_plus', 'dq', 'sq', 'pq'] 
                # metrics = evalute_overall_im(dataset_name, model_name, pred_dir, model,  type_uid_list=list(range(num_classes)))
                # metrics = evalute_overall_im(dataset_name, model_name, pred_dir, model,  type_uid_list=list(range(num_classes)))
                # (map50, map75), (acc, f1, *f), (dice, aji, ajip, dq, sq, pq) = metrics
                
                
            
            # early stopping
            early_stopping(valid_logs[key], model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
            # wandb log
            # 如果选择了不同的loss，这里的key会有变化
            wandb.log({
                'train_loss': train_logs[key],
                'val_loss': valid_logs[key],
                'train_iou': train_logs['iou_score'],
                'val_iou': valid_logs['iou_score'],
                # 'metrics': metrics, #log 多个元素ok么
                # # 'stop_epoch':i,
                # 'map50' :map50.item(),
                # 'map75':map75.item(),
                # 'acc':acc,
                # 'f1':f1,
                # 'dice':dice,
                # 'aji':aji,
                # 'ajip':ajip, 
                # 'dq':dq,
                # 'sq':sq,
                # 'pq':pq
            })
        except:
            print("something wrong!")
        
        
        
def train_profiler(config, num_classes, model_name, dataset_name, dir_root, log_dir):
    epochs = config.epochs
    rm_n_mkdir(log_dir)
    train_dataset = PanNukeDataset(
                        images_dir = f'{dir_root}/images/train', 
                        masks_dir = f'{dir_root}/seg_mask/train/', 
                        num_classes = num_classes,
                        augmentation=aug_fn(), 
                        preprocessing=process_fn(),
                        )

    valid_dataset = PanNukeDataset(
                        images_dir= f'{dir_root}/images/test',
                        masks_dir = f'{dir_root}/seg_mask/test', 
                        num_classes=num_classes,
                        augmentation=None,          # test 的时候需要使用 augmentation 么？暂定不用
                        preprocessing=process_fn(),
                    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = smp.Unet(
                encoder_name=config.encoder_name,        # choose encoder, e.g. resnet34, mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=num_classes,            # model output channels (number of classes in your dataset)
                activation=config.activation
                )

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    if config.loss == "DiceLoss":
        loss = losses.DiceLoss()
    elif config.loss == "JaccardLoss":
        loss = losses.JaccardLoss()
    elif config.loss == "FocalLoss":
        loss = losses.FocalLoss()
    elif config.loss == "SoftCrossEntropyLoss":
        loss = losses.SoftCrossEntropyLoss()
    else:
        print("loss setting is wrong!")

    # 对比最后的评估，这里有内置的评估
    metrics = [
        IoU(threshold=0.5),
    ]

    if config.optimizer == "adam":
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=config.lr)])
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD([dict(params=model.parameters(), lr=config.lr)])
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW([dict(params=model.parameters(), lr=config.lr)])
    else:
        print("optimizer setting is wrong!")

    # 初始化 early_stopping 对象
    patience = 7	# 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=True, delta=0.01)	# 关于 EarlyStopping 的代码可先看博客后面的内容


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
#         model = model.to('cuda')
#         train_logs = train_epoch.run(train_loader)
#         valid_logs = valid_epoch.run(valid_loader)
        
#         # do something (save model, change lr, etc.)
#         if max_score < valid_logs['iou_score']:
#             max_score = valid_logs['iou_score']
#             torch.save(model, f'{log_dir}/model_{i}.pth')
#             print('Model saved!')
        
        if i % config.val_interval == 0:
            dataset_name = dataset_name
            model_name = "seg_unet"
            pred_dir = f"/root/autodl-tmp/datasets/{dataset_name}/images/test"
            
            
            # metrics = evalute_overall_im(dataset_name, model_name, pred_dir, model,  type_uid_list=list(range(num_classes)))
            
        
#         # early stopping
#         early_stopping(valid_logs['dice_loss'], model)
#         # 若满足 early stopping 要求
#         if early_stopping.early_stop:
#             print("Early stopping")
#             # 结束模型训练 epoch
#             break
            
        # wandb log
        # wandb.log({
        #     'train_loss': train_logs['dice_loss'],
        #     'val_loss': valid_logs['dice_loss'],
        #     'mertic_iou': valid_logs['iou_score'],
        #     'metrics': metrics,
        #      'stop_epoch':i
        # })