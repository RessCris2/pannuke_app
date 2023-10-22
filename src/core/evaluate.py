
from os.path import join as opj
from tqdm import tqdm
import numpy as np
import scipy.io  as sio
import pandas as pd

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
torch.cuda.empty_cache()
# import matplotlib.pyplot as plt
import sys
from src.core.utils import  fn_time
from src.core.stats_utils import eveluate_one_pic_class, eveluate_one_pic_inst, run_nuclei_type_stat
from src.core.infer_base import predict_oneimg, predict_dir, hovernet_predict_dir
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import time
# from line_profiler import profile
from memory_profiler import profile

# @fn_time
def evalute_oneimg(img_path, dataset_name, model_name, model_path_dict):
    """重点可能在于处理 GT 的部分
    """
    pred, true = predict_oneimg(img_path, dataset_name, model_name, model_path_dict)
    preds = [dict(
        boxes=torch.tensor(pred['bboxes']),
        scores=torch.tensor(pred['scores']),
        labels=torch.tensor(pred['labels']),
        masks =torch.tensor(pred['masks'], dtype=torch.uint8),
    )]

    
    target = [dict(
        boxes=torch.tensor(true['bboxes']),
        scores=torch.tensor(true['scores']),
        labels=torch.tensor(true['labels']),
        masks =torch.tensor(true['masks'], dtype=torch.uint8),
    )]

    metric = MeanAveragePrecision(iou_type='segm')
    metric.update(preds, target)
    metric0 = metric.compute()
    from pprint import pprint
    # pprint(metric0)


    # evaluate class
    metric1 = eveluate_one_pic_class(true['centroids'], pred['centroids'], true['labels'], pred['labels'])
    # print(metric1)
    # evaluate inst
    metric2 = eveluate_one_pic_inst(true['masks'], pred['masks'])
    return metric0, metric1, metric2


# @fn_time
def evalute_oneimg_v2(pred, true, metric2=None):
    """重点可能在于处理 GT 的部分
    """
    # predcit_fn  = "{}_predict_oneimg".format(model_name)
    # pred = eval(predcit_fn)(img_path, dataset_name, model_name)
    # true = load_gt(img_path)
    
    # evaluate map
    preds = [dict(
        boxes=torch.tensor(pred['bboxes']),
        scores=torch.tensor(pred['scores']),
        labels=torch.tensor(pred['labels']),
        masks =torch.tensor(pred['masks'], dtype=torch.uint8),
    )]


    target = [dict(
        boxes=torch.tensor(true['bboxes']),
        scores=torch.tensor(true['scores']),
        labels=torch.tensor(true['labels']),
        masks =torch.tensor(true['masks'], dtype=torch.uint8),
    )]

    metric = MeanAveragePrecision(iou_type='segm')
    metric.update(preds, target)
    metric0 = metric.compute()
    metric0 = pd.DataFrame(metric0, index=[0], columns=['map_50', 'map_75']).values[0].tolist()
    from pprint import pprint
    # pprint(metric0)


    # evaluate class
    metric1 = eveluate_one_pic_class(true['centroids'], pred['centroids'], true['labels'], pred['labels'])
    # print(metric1)
    # evaluate inst
    if metric2 is not None:
        return np.concatenate([metric0, metric1, metric2])
    
    else:
        metric2 = eveluate_one_pic_inst(true['masks'], pred['masks'])
        return np.concatenate([metric0, metric1, metric2])


def evalute_overall(dataset_name, model_name, model_path_dict, ex_name = None, is_detail=True, type_uid_list=[0, 1, 2, 3, 4]):
    """
    ex_name : 标记数据集名称
    """
    pred_dir = "/root/autodl-tmp/archive/datasets/{}/patched/coco_format/images/test".format(dataset_name)

    if model_name == 'hovernet':
        img_names, preds_result, trues_result = hovernet_predict_dir(pred_dir, dataset_name, model_name, model_path_dict)
    else:
        img_names, preds_result, trues_result = predict_dir(pred_dir, dataset_name, model_name, model_path_dict)
  
    def concat_res(result):
        processed = []
        for res in result:
            ress = dict(
            boxes=torch.tensor(res['bboxes']),
            scores=torch.tensor(res['scores']),
            labels=torch.tensor(res['labels']),
            masks =torch.tensor(res['masks'], dtype=torch.uint8)
             )
            processed.append(ress)
        return processed
    
    preds = concat_res(preds_result)
    trues = concat_res(trues_result)


    metric = MeanAveragePrecision(iou_type='segm')
    metric.update(preds, trues)
    metrics0 = metric.compute()
    metrics0 = pd.DataFrame(metrics0, index=[0], columns=['map_50', 'map_75']).values[0].tolist()
    # pprint(metrics0)

    # 处理 class 部分; 作为 overall 结果处理的。
    metrics1 = run_nuclei_type_stat(preds_result, trues_result, type_uid_list)
   

    if is_detail:
        detail_result = []
        start = time.time()
        detail_result = []
        with ProcessPoolExecutor(max_workers=4) as executor:
            ## data 为每个 augument 函数的返回值。
            for data in executor.map(evalute_oneimg_v2, preds_result, trues_result):
                detail_result.append(data)

        elapsed_time = time.time() - start
        print("计算每张图片的表现共耗时{}s".format(elapsed_time))
        metrics2 = np.mean(np.array(detail_result)[: ,-6:], axis=0)
        # TODO: 待验证
        detail_result = pd.DataFrame(detail_result, columns=['map_50', 'map_75', 'acc', 'f1', 'dice','aji', 'aji_plus', 'dq', 'sq', 'pq'])
        detail_result['img_names'] = img_names
        if ex_name is not None:
            detail_result.to_csv("/root/autodl-tmp/archive/core/metrics/evaluate_gen/{}_{}_{}_detail.csv".format(dataset_name, model_name, ex_name))
        else:
            detail_result.to_csv("/root/autodl-tmp/archive/core/metrics/evaluate_result/{}_{}_detail_p2.csv".format(dataset_name, model_name))
    
    # metrics2 = np.mean(detail_result[-6:], axis=0)
    columns = ['map_50', 'map_75'] + ['acc', 'f1'] + [str(i) for i in type_uid_list] + ['dice','aji', 'aji_plus', 'dq', 'sq', 'pq'] 
    overall_result = pd.DataFrame(np.concatenate([metrics0, metrics1, metrics2])[None, :], columns=columns)

    if ex_name is not None:
        overall_result.to_csv("/root/autodl-tmp/archive/core/metrics/evaluate_gen/{}_{}_{}_overall.csv".format(dataset_name, model_name, ex_name))
    else:
        overall_result.to_csv("/root/autodl-tmp/archive/core/metrics/evaluate_result/{}_{}_overall_p2.csv".format(dataset_name, model_name))
    
    if ex_name is None:
        return overall_result

@fn_time
def evalute_overall_im(dataset_name, model_name, pred_dir, model, is_detail=True, type_uid_list=[0, 1, 2, 3, 4]):
    """
    ex_name : 标记数据集名称
        测试的时候时候的版本，但是其实只是为了处理一下 model_path_dict
    """
    if model_name == 'hovernet':
        img_names, preds_result, trues_result = hovernet_predict_dir(pred_dir, dataset_name, model_name, model_path_dict=None)
    else:
        img_names, preds_result, trues_result = predict_dir(pred_dir, dataset_name, model_name, model_path_dict=None, model=model, load_model=False)
  
    def concat_res(result):
        processed = []
        for res in result:
            ress = dict(
            boxes=torch.tensor(res['bboxes']),
            scores=torch.tensor(res['scores']),
            labels=torch.tensor(res['labels']),
            masks =torch.tensor(res['masks'], dtype=torch.uint8)
             )
            processed.append(ress)
        return processed
    
    preds = concat_res(preds_result)
    trues = concat_res(trues_result)


    metric = MeanAveragePrecision(iou_type='segm')
    metric.update(preds, trues)
    metrics0 = metric.compute()
    metrics0 = pd.DataFrame(metrics0, index=[0], columns=['map_50', 'map_75']).values[0].tolist()
    # pprint(metrics0)

    # 处理 class 部分; 作为 overall 结果处理的。
    try:
        metrics1 = run_nuclei_type_stat(preds_result, trues_result, type_uid_list)
    except:
        print("class result is None")
        metrics1=(None, None, None)

    if is_detail:
        detail_result = []
        start = time.time()
        detail_result = []
        with ProcessPoolExecutor(max_workers=4) as executor:
            ## data 为每个 augument 函数的返回值。
            for data in executor.map(evalute_oneimg_v2, preds_result, trues_result):
                detail_result.append(data)

        elapsed_time = time.time() - start
        print("计算每张图片的表现共耗时{}s".format(elapsed_time))
        metrics2 = np.mean(np.array(detail_result)[: ,-6:], axis=0)
        # TODO: 待验证
        # detail_result = pd.DataFrame(detail_result, columns=['map_50', 'map_75', 'acc', 'f1', 'dice','aji', 'aji_plus', 'dq', 'sq', 'pq'])
        # detail_result['img_names'] = img_names
       
    # metrics2 = np.mean(detail_result[-6:], axis=0)
    columns = ['map_50', 'map_75'] + ['acc', 'f1'] + [str(i) for i in type_uid_list] + ['dice','aji', 'aji_plus', 'dq', 'sq', 'pq'] 
    # overall_result = pd.DataFrame(np.concatenate([metrics0, metrics1, metrics2])[None, :], columns=columns)
    return metrics0, metrics1, metrics2
    
    
def batch_tasks():
    ## 批量跑任务, 给结果数据加上 img basename
    from eval_config import tasks, model_path_dict
    for task in tasks:
        start_t = time.time()
        evalute_overall(task['dataset_name'], task['model_name'], model_path_dict)
        print(task['dataset_name'], task['model_name'],time.time()-start_t)

def batch_tasks_old():
        ## 批量跑任务
    tasks = pd.read_excel('/root/autodl-tmp/archive/core/metrics/evaluate_tasks.xlsx')
    eval_tasks = tasks.query("is_enable == 1")[['dataset_name', 'model_name', 'predict_fn']].to_dict('records')

    # executor = ProcessPoolExecutor(max_workers=10)
    for task in eval_tasks:
        task.update({"pred_dir": "/root/autodl-tmp/archive/datasets/{}/patched/coco_format/images/test".format(task['dataset_name'])})
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        
        for idx, e in  zip( range(len(eval_tasks)), executor.map(evalute_overall, [task['pred_dir'] for task in eval_tasks],
                            [task['predict_fn'] for task in eval_tasks],
                            [task['dataset_name'] for task in eval_tasks],
                            [task['model_name'] for task in eval_tasks])):
                print(" {} is running!".format(idx))



def evaluate_type(dataset_name, model_name, model_path_dict, type_uid_list=[0, 1, 2, 3, 4]):
    """
    ex_name : 标记数据集名称
    """
    pred_dir = "/root/autodl-tmp/archive/datasets/{}/patched/coco_format/images/test".format(dataset_name)

    if model_name == 'hovernet':
        img_names, preds_result, trues_result = hovernet_predict_dir(pred_dir, dataset_name, model_name, model_path_dict)
    else:
        img_names, preds_result, trues_result = predict_dir(pred_dir, dataset_name, model_name, model_path_dict)
  
    def concat_res(result):
        processed = []
        for res in result:
            ress = dict(
            boxes=torch.tensor(res['bboxes']),
            scores=torch.tensor(res['scores']),
            labels=torch.tensor(res['labels']),
            masks =torch.tensor(res['masks'], dtype=torch.uint8)
             )
            processed.append(ress)
        return processed
    
    preds = concat_res(preds_result)
    trues = concat_res(trues_result)

    metrics1 = run_nuclei_type_stat(preds_result, trues_result, type_uid_list)
  
    # columns = ['map_50', 'map_75'] + ['acc', 'f1'] + [str(i) for i in type_uid_list] + ['dice','aji', 'aji_plus', 'dq', 'sq', 'pq'] 
    # overall_result = pd.DataFrame(np.concatenate([metrics0, metrics1, metrics2])[None, :], columns=columns)

    
    return metrics1


        
if __name__ == "__main__":
    # dataset_name = 'consep'
    # model_name = 'mask_rcnn'
    # pred_dir = "/root/autodl-tmp/archive/datasets/{}/patched/coco_format/images/test".format(dataset_name)
    # predict_fn = 'mask_rcnn_predict_dir'
    # batch_tasks()
    pass
