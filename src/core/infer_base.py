"""是否可以统一所有模型的预测过程？
    这里应该就是可以做一个类似于接口的工作？在 python 中如何写接口？
"""
import argparse
import cProfile as profile
import os
import pathlib
from os.path import join as opj

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from tqdm import tqdm

torch.cuda.empty_cache()
import sys

import matplotlib.pyplot as plt
from memory_profiler import profile
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules
## mmseg 部分代码
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot
from scipy.ndimage import label
from src.dist.dataloader import data_aug
## dist 部分代码
from src.dist.dist_net import DIST

from src.core.post_proc import post_process
from src.core.pre_proc import process_fn
from src.core.utils import (fetch_inst_centroid_maskrcnn, find_files, fn_time,
                            get_bounding_box, get_curtime, load_gt, load_img,
                            loads_model, rm_n_mkdir, transfer_inst_format)
## hovernet
from src.hovernet.infer.tile_pannuke import InferManager


def loads_model_dynamic(model_name, dataset_name,  model_path_dict):
    if model_name in ['mask_rcnn', 'cascade_rcnn']:
        config_file = model_path_dict[dataset_name][model_name]['config_file']
        checkpoint_file = model_path_dict[dataset_name][model_name]['checkpoint_file']
        register_all_modules()
        model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cpu'
        return model
    elif model_name =="dist":
        save_path = model_path_dict[dataset_name][model_name]['save_path']
        model = DIST(num_features=6)
        model = loads_model(model, save_path)
        return model
    elif model_name == 'seg_unet':
        model_path = model_path_dict[dataset_name][model_name]['model_path']
        model = torch.load(model_path, map_location='cpu')
        return model
    elif model_name == 'mmseg_fcn':
        config_file = model_path_dict[dataset_name][model_name]['config_file']
        checkpoint_file = model_path_dict[dataset_name][model_name]['checkpoint_file']
        model = init_model(config_file, checkpoint_file, device='cuda:0')
        if not torch.cuda.is_available():
            model = revert_sync_batchnorm(model)
        return model
    else:
        print('No matched models')

@fn_time
def mask_rcnn_predict_oneimg(img_path, dataset_name, model_name, model_path_dict, model=None, load_model=True, score_thr=0.3, ):
    """
        load_model: 如果要直接传入数据 [config_file, checkpoint_file], 修改 config_file 才行？
    """
    if load_model:
        model = loads_model_dynamic(model_name, dataset_name, model_path_dict)
    result = inference_detector(model, img_path) ## 注意这里，可以使用 img_path, 不需要自己加载
    res = result.pred_instances.to_dict()
    pred = dict()
    pred['labels'] = res['labels'][res['scores']>score_thr].cpu().numpy()
    pred['masks'] = res['masks'][res['scores']>score_thr].cpu().numpy().astype(int)
    pred['bboxes'] = res['bboxes'][res['scores']>score_thr].cpu().numpy().astype(int)
    pred['scores'] = res['scores'][res['scores']>score_thr].cpu().numpy()
    pred['centroids']= fetch_inst_centroid_maskrcnn(pred['masks'])

    if len(pred['labels']) == 0:
        raise ValueError

    true = load_gt(img_path)
    return pred, true


@fn_time
def dist_predict_oneimg( img_path, dataset_name, model_name, model_path_dict, model=None, load_model=True):
    if load_model:
        model = loads_model_dynamic(model_name, dataset_name, model_path_dict)
    img = load_img(img_path)
    img = data_aug()(img).unsqueeze(0)
    mp = model(img).detach().numpy().squeeze()
    mp = np.clip(mp, 0, 255 )
    pred_inst = post_process(mp) #  p1, p2 取默认值
    pred = transfer_inst_format(pred_inst, true_type_map=None)
    true = load_gt(img_path, is_class=False)
    return pred, true


def dist_predict_oneimg_for_plot( img_path, dataset_name, model_name, model_path_dict, model=None, load_model=True):
    if load_model:
        model = loads_model_dynamic(model_name, dataset_name, model_path_dict)
    img = load_img(img_path)
    img = data_aug()(img).unsqueeze(0)
    mp = model(img).detach().numpy().squeeze()
    mp = np.clip(mp, 0, 255 )
    pred_inst = post_process(mp) #  p1, p2 取默认值
    # pred = transfer_inst_format(pred_inst, true_type_map=None)
    # true = load_gt(img_path, is_class=False)
    return pred_inst


# @fn_time
def seg_predict_oneimg(img_path, dataset_name,  model_name, model_path_dict, model=None, load_model=True, device = 'cpu'):
    if load_model:
        model = loads_model_dynamic(model_name, dataset_name, model_path_dict)

    image = load_img(img_path)
    image = process_fn()(image)
    x_tensor = image.unsqueeze(0)
    with torch.no_grad():
        res = model.to(device).predict(x_tensor.to(device)).squeeze().detach().numpy()
    # prob_map = res[1:].sum(axis=0)  # 这里可能需要纠正模型的类别对应问题。res or  res[1:]
    # prob_map = res[1:].max(axis=0)
    pred_labels = np.argmax(res, axis=0)
    # dist 同样的 后处理会无法调节参数
    # pred = post_process(prob_map, mode='prob')
    pred = label(pred_labels)[0]
    
    # type_map = np.argmax(res, axis=0) # 获取 type_map
    pred = transfer_inst_format(pred, true_type_map=pred_labels)
    true = load_gt(img_path)
    return pred, true


def seg_predict_oneimg_for_plot(img_path, dataset_name,  model_name, model_path_dict, model=None, load_model=True, device = 'cpu'):
    if load_model:
        model = loads_model_dynamic(model_name, dataset_name, model_path_dict)

    image = load_img(img_path)
    image = process_fn()(image)
    x_tensor = image.unsqueeze(0)
    with torch.no_grad():
        res = model.to(device).predict(x_tensor.to(device)).squeeze().detach().numpy()
    # prob_map = res[1:].sum(axis=0)  # 这里可能需要纠正模型的类别对应问题。res or  res[1:]
    # prob_map = res[1:].max(axis=0)
    pred_labels = np.argmax(res, axis=0)
    # dist 同样的 后处理会无法调节参数
    # pred = post_process(prob_map, mode='prob')
    pred = label(pred_labels)[0]
    
    # type_map = np.argmax(res, axis=0) # 获取 type_map
    # pred = transfer_inst_format(pred, true_type_map=pred_labels)
    # true = load_gt(img_path)
    # return pred, true
    return pred_labels, pred

@fn_time
def mmseg_predict_oneimg(img_path, dataset_name, model_name, model_path_dict, model=None, load_model=True, thresh=0.9):
    if load_model:
        model = loads_model_dynamic(model_name, dataset_name, model_path_dict)
    result = inference_model(model, img_path)
    re =result.to_dict()
    pred_labels = re['pred_sem_seg']['data'][0].cpu().detach().numpy()
    labels = label(pred_labels)[0]
    pred = transfer_inst_format(labels, true_type_map=pred_labels)
    true = load_gt(img_path)
    return pred, true
    # return pred


def hovernet_predict_dir(pred_dir, dataset_name, model_name, model_path_dict ):
    # run_args
    # input_dir= "/root/autodl-tmp/archive/datasets/{}/patched/coco_format/images/test".format(dataset_name)
    
    # method_args
    if dataset_name == 'pannuke':
        nr_types = 6
    elif dataset_name in ['monusac', 'consep']:
        nr_types = 5
    elif dataset_name in ['kumar']:
        nr_types = None
    else:
        print('dataset is not right!')
        raise ValueError

    method_args = {
        'method': {
            'model_args': {
                'mode': 'fast',
                'nr_types': nr_types,
            },
            'model_path':model_path_dict[dataset_name][model_name]['model_path']
        },
    }

    run_args={
        'input_dir': pred_dir,
        'nr_inference_workers': 4,
        'batch_size': 4,
        'nr_post_proc_workers':2,
        'dataset_name': dataset_name
    }
    infer = InferManager(**method_args)
    # 如何处理对应的问题呢？打印出 img_name 最好。
    img_names, preds = infer.process_file_list(run_args)
    
    trues = []
    
    if os.path.isdir(pred_dir):
        for name in img_names:
            img_path = opj(pred_dir, '{}.jpg'.format(name))
            true = load_gt(img_path, is_class=True)
            trues.append(true)
             # 后续算法也应该做这样的调整，保留一下数据的 basename
            return img_names, preds, trues
    else:
        img_path = pred_dir
        true = load_gt(img_path, is_class=True)
        # trues.append(true)
        pred = preds[0]
        # img_names = img_names[0]
        return pred, true


def predict_oneimg(file, dataset_name, model_name, model_path_dict, model=None,  load_model=True):
    if model_name in ['mask_rcnn', 'cascade_rcnn']:
        pred, true = mask_rcnn_predict_oneimg(file, dataset_name, model_name, model_path_dict, model, load_model)
        return pred, true
    elif model_name =="dist":
        pred, true = dist_predict_oneimg(file, dataset_name, model_name, model_path_dict, model, load_model)
        return pred, true
    elif model_name == 'seg_unet':
        pred, true = seg_predict_oneimg(file, dataset_name, model_name, model_path_dict, model, load_model)
        return pred, true
    elif model_name == 'mmseg_fcn':
        pred, true = mmseg_predict_oneimg(file, dataset_name, model_name, model_path_dict, model, load_model)
        return pred, true
    elif model_name == 'hovernet':
        pred, true = hovernet_predict_dir(file, dataset_name, model_name, model_path_dict)
        return pred, true
    else:
        print('No matched models')    

# @profile
def predict_dir(pred_dir, dataset_name, model_name, model_path_dict=None, model=None, load_model=True, is_exs=True):
    # pred_dir = "/root/autodl-tmp/archive/datasets/{}/patched/coco_format/images/test".format(dataset_name)
    if load_model:
        model = loads_model_dynamic(model_name, dataset_name, model_path_dict)
    torch.cuda.empty_cache()
    if is_exs:
        exs = "/root/autodl-tmp/src/core/exs/{}_exs.npy".format(dataset_name)
    else:
        exs=None
    files = find_files(pred_dir, ext='jpg', exs=exs)
    preds = []
    trues = []
    length = 5
    basenames = []
    for file in files[:length]:
        try:
            pred, true = predict_oneimg(file, dataset_name, model_name, model_path_dict, model, load_model=False)
            preds.append(pred)
            trues.append(true)
            basename = pathlib.Path(file).stem
            basenames.append(basename)
        except:
            continue

    return basenames, preds, trues

if __name__ == "__main__":
    # 验证几个函数的正确性。
    # 第一个 mask_rcnn_predict_oneimg
    dataset_name = "consep"
    model_name  = 'mmseg_fcn'
    # predcit_fn  = "{}_predict_oneimg".format(model_name)
    # 2 seg_fcn_predict_oneimg
    predcit_fn = 'mmseg_predict_dir'

    pred_dir = "/root/autodl-tmp/archive/datasets/{}/patched/coco_format/images/test".format(dataset_name)
    img_path = find_files(pred_dir, ext='jpg')[0]
    pred = eval(predcit_fn)(pred_dir, dataset_name, model_name)
    print('xxx')