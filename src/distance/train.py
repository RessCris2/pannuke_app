"""
  训练 UNetDIST 模型
"""
import time
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from  os.path import join as opj
import sys
from tqdm import tqdm
import numpy as np
import scipy.io as sio
import logging

# relative import
from .dist_net import DIST, loss_fn
from .dataloader import DISTDataset
from .predict import predict
from ..core.utils import get_logger, rm_n_mkdir, loads_model, load_img, find_files, get_curtime

############## TENSORBOARD ########################
import sys
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import wandb
from torch.cuda.amp import autocast
# default `log_dir` is "runs" - we'll be more specific here
# import sys
# sys.path.append("/root/autodl-tmp/archive/v2/metrics")
# from evaluate import evalute_oneimg_v2
# from infer import dist_predict_oneimg

###################################################

# def evaluate_one_pic(model ):
#     """只针对一张图片计算指标结果
#     """
#     path = "/root/autodl-tmp/datasets/pannuke/inst/test/0.npy"
#     true = np.load(path)
#     # 只对一张图片进行评估
#     img = load_img("/root/autodl-tmp/datasets/pannuke/coco_format/images/test/0.jpg")
#     pred = predict(model, img)
#     metrics = run_one_inst_stat(true, pred, match_iou=0.5)
#     return pred, metrics


def train(dataset_name, dir_root, save_dir, model_name='dist'):
    epochs = 100
    batch_size = 4
    val_interval = 2
    save_interval = 5
    num_features = 6  # 这个参数可以调节模型的复杂度
    
    rm_n_mkdir(save_dir)
    logger = get_logger(log_file_name='train.log', log_dir=save_dir)
    
    writer_save_dir = os.path.join(f"{save_dir}/tf_logs")
    rm_n_mkdir(writer_save_dir)
    writer = SummaryWriter(writer_save_dir)

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = dict()
    classes.update({'pannuke':['Background','Neoplastic','Inflammatory','Connective','Dead','Epithelial']})
    classes.update({'monusac': ['Background','Epithelial','Lymphocyte','Neutrophil','Macrophage']})
    classes.update({'consep': ['background','inflammatory','healthy_epithelial','epithelial', 'spindle-shaped']})
    classes.update({'cpm17':None}) # 处理一下分类数据为 None 的情况，应该是读入的时候就不一样

    # 训练集
    train_images_dir = opj(dir_root, 'images/train')
    train_masks_dir =  opj(dir_root, 'seg_mask/train')
    train_loader = DataLoader(dataset=DISTDataset(train_images_dir, train_masks_dir, classes=classes[dataset_name]),
                              batch_size = batch_size,
                              shuffle=True,
                              drop_last=False,
                              num_workers=8)
    # 测试机
    # dir_root = "/root/autodl-tmp/datasets/pannuke/coco_format/"
    test_images_dir = opj(dir_root, 'images/test')
    test_masks_dir =  opj(dir_root, 'seg_mask/test')
    test_loader = DataLoader(dataset=DISTDataset(test_images_dir, test_masks_dir, classes=classes[dataset_name]),
                              batch_size = batch_size,
                              shuffle=False,
                              drop_last=False,
                              num_workers=8)


  
    # example_data = []
    # for iter, (image, label, img_path) in enumerate(test_loader):
    #     if iter < 6:
    #         image, label = image.to(device), label.to(device)
    #         example_data.append(image[0])
    #     else:
    #         break

    ############## TENSORBOARD ########################
    # img_grid = torchvision.utils.make_grid(example_data)
    # writer.add_image('test_images', img_grid)
    
    ###################################################

    model = DIST(num_features=num_features).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=0.00001)
    
    ############## TENSORBOARD ########################
    # writer.add_graph(model, example_data.to(device))
    #writer.close()
    #sys.exit()
    ###################################################

    n_total_steps = len(train_loader)
    for epoch in tqdm(range(epochs)):
        start = time.perf_counter() 
        model = model.to(device)
        loss_train = 0
        for iter_idx, (image, label, _) in enumerate(train_loader):
            model.train()
            image, label = image.to(device), label.to(device)
            pred = model(image) 
            loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss
        
        
        ############## TENSORBOARD ########################
        # writer.add_scalar('training loss', loss, epoch * n_total_steps + iter_idx)
        # writer.add_scalar('training loss', loss_train, epoch)
        # running_accuracy = running_correct / 100 / predicted.size(0)
        # writer.add_scalar('accuracy', running_accuracy, epoch * n_total_steps + i)
        # running_correct = 0
        # running_loss = 0.0
        ###################################################

        # 每隔保存 interval 保存模型
        if epoch % save_interval == 0:
            save_path = opj(save_dir, "epoch_{}.pth".format(epoch))
            torch.save(model.state_dict(), save_path)

        # 每隔保存 interval 保存模型
        n_test_steps = len(test_loader)
        if epoch % val_interval == 0:
            loss_item = 0
            # metrics_item = []
            for iter_idx, (image, label, img_paths) in enumerate(test_loader):
                model.eval()
                model = model.cpu()
                #这个部分需要模型是 cuda
                # 或者这里改为 cpu
                image, label = image.to('cpu'), label.to('cpu')
                pred = model(image) 
                loss = loss_fn(pred, label)
                loss_item += loss

            #     for img_path in img_paths:
            #         try:
            #             # 这个部分需要模型是 cpu
            #             pred, true = dist_predict_oneimg( img_path, dataset_name, model_name, model=model, load_model=False)
            #             metrics = evalute_oneimg_v2(pred, true)[-8:]
            #             metrics_item.append(metrics)
            #         except:
            #             continue
            #         # logger.info("metrics 为 {}".format(metrics))
            # metrics_item = np.mean(metrics_item, axis=0)
            
            # writer.add_scalar('validataion loss', loss_item, epoch)
            # writer.add_scalar('acc', metrics_item[0], epoch)
            # writer.add_scalar('f1_score', metrics_item[1], epoch)
            # writer.add_scalar('dice', metrics_item[2], epoch)
            # writer.add_scalar('aji', metrics_item[3], epoch)
            # writer.add_scalar('aji_plus', metrics_item[4], epoch)
            # writer.add_scalar('dq', metrics_item[5], epoch)
            # writer.add_scalar('sq', metrics_item[6], epoch)
            # writer.add_scalar('pq', metrics_item[7], epoch)

            # 每隔多少个 epoch 就对val datast 进行评估。记录几个指标的变化

        dur = time.perf_counter() - start    # 计时，计算进度条走到某一百分比的用时
        logger.info("epoch_{} 耗费时间为{}s".format(epoch, dur))

def train_wb(config, dataset_name, dir_root, save_dir, model_name='dist'):
    # setting up 
    torch.cuda.empty_cache()
    rm_n_mkdir(save_dir)
    logger = get_logger(log_file_name='train.log', log_dir=save_dir)
    writer_save_dir = os.path.join(f"{save_dir}/tf_logs")
    rm_n_mkdir(writer_save_dir)
    writer = SummaryWriter(writer_save_dir)

    classes = dict()
    classes.update({'pannuke':['Background','Neoplastic','Inflammatory','Connective','Dead','Epithelial']})
    classes.update({'monusac': ['Background','Epithelial','Lymphocyte','Neutrophil','Macrophage']})
    classes.update({'consep': ['background','inflammatory','healthy_epithelial','epithelial', 'spindle-shaped']})
    classes.update({'cpm17':None}) 

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # build dataloader
    ## train set
    train_images_dir = opj(dir_root, 'images/train')
    train_masks_dir =  opj(dir_root, 'seg_mask/train')
    train_loader = DataLoader(dataset=DISTDataset(train_images_dir, train_masks_dir, classes=classes[dataset_name]),
                              batch_size = config.batch_size,
                              shuffle=True,
                              drop_last=False,
                              num_workers=config.num_workers)
    ## test set
    test_images_dir = opj(dir_root, 'images/test')
    test_masks_dir =  opj(dir_root, 'seg_mask/test')
    test_loader = DataLoader(dataset=DISTDataset(test_images_dir, test_masks_dir, classes=classes[dataset_name]),
                              batch_size = config.batch_size,
                              shuffle=False,
                              drop_last=False,
                              num_workers=1)
    # build model
    model = DIST(num_features=config.num_features).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # train model
    n_total_steps = len(train_loader)
    for epoch in tqdm(range(config.epochs)):
        start = time.perf_counter() 
        model = model.to(device)
        train_loss = 0
        for iter_idx, (image, label, _) in enumerate(train_loader):
            model.train()
            image, label = image.to(device), label.to(device)
            # with autocast():
            pred = model(image) 
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        # save models every val_interval
        # n_test_steps = len(test_loader)
        if epoch % config.val_interval == 0:
            with torch.no_grad():
                validation_loss = 0
                for iter_idx, (image, label, img_paths) in enumerate(test_loader):
                    model.eval()
                    image, label = image.to(device), label.to(device)
                    pred = model(image)
                    loss = loss_fn(pred, label)
                    validation_loss += loss.item()
                
        # save models every save_interval
        if epoch % config.save_interval == 0:
            save_path = opj(save_dir, "epoch_{}.pth".format(epoch))
            torch.save(model.state_dict(), save_path)
        
        wandb.log({'train_loss': train_loss,
              'val_loss': validation_loss
              })
    # 少了一个评估结果, 就是几个指标
    

    # 统计整个训练的时长
    dur = time.perf_counter() - start    
    logger.info("epoch_{} takes {}s".format(epoch, dur))
    wandb.log({'dur': dur })
    del train_loss
    del validation_loss
    # return train_loss, validation_loss

if __name__ == "__main__":
    dataset_name='consep'
    model_name = 'dist'
    train(dataset_name, model_name)
    # model = DIST(num_features=6)
    # model = load_model(model, path = "/root/autodl-tmp/com_models/dist_torch/model_data/epoch_10.pth")
    # pred, _ = evaluate_one_pic(model)
