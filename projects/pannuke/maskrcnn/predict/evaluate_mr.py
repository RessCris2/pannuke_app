"""不做额外的 mAP 计算工作
只评估 dice, aji
"""
import json
import os
import sys

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append("/root/autodl-tmp/pannuke_app")
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from src.evaluation.stats_utils_v2 import eveluate_one_pic_inst

sys.path.append("/root/autodl-tmp/pannuke_app")
from src.evaluation.evaluate import calculate_map, evaluate
from src.evaluation.stats_utils_v2 import eveluate_one_pic_inst


def evaluate_map(ann_file, pred_result_dir, save_path=None):
    test = COCO(ann_file)
    results = []
    metrics = []
    basenames = []
    for image in test.dataset["images"]:
        basename = os.path.basename(image["file_name"])
        file_name = "{}/{}".format(
            pred_result_dir, image["file_name"].replace("png", "json")
        )
        with open(file_name, "r") as f:
            result = json.load(f)
        result["image_id"] = image["id"]

        bboxs = []
        for bbox in result["bboxes"]:
            x_min, y_min, x_max, y_max = bbox
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            bbox = [x_center, y_center, width, height]
            bboxs.append(bbox)

        result["bbox"] = bboxs

        result["category_id"] = result.pop("labels")
        result["segmentation"] = result.pop("masks")
        result["score"] = result.pop("scores")

        # 将 result 中的列表格式改为 字典格式，方便 coco.loadRes
        # trans = []
        items = []
        pred_masks = []
        for d1, d2, d3, d4 in zip(
            result["bbox"],
            result["category_id"],
            result["segmentation"],
            result["score"],
        ):
            item = {}
            item.update({"bbox": d1})
            item.update({"category_id": d2})
            item.update({"segmentation": d3})
            item.update({"score": d4})
            item.update({"image_id": image["id"]})
            items.append(item)
            mask = mask_utils.decode(d3)
            pred_masks.append(mask)

        results.extend(items)
        # 每张图的多个实例mask
        # print(pred_masks)
        # gt_masks = []
        # # 可以从 coco 的 ann 中获取 gt_masks
        # anns = test.loadAnns(test.getAnnIds(imgIds=[image["id"]]))
        # for ann in anns:
        #     gt_masks.append(test.annToMask(ann))
        # # print(gt_masks)
        # metric = eveluate_one_pic_inst(gt_masks, pred_masks)
        # metrics.append(metric)
        # basenames.append(basename)
    # print(metrics)
    overall_map, map_pd = calculate_map(results, ann_file, save_path)
    # test = COCO(ann_file)
    # tt = test.loadRes(results)
    # cocoEval = COCOeval(test, tt, iouType="segm")
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()
    # overall_map = cocoEval.stats[:2]

    # per_image_mAPs = []
    # basenames = []
    # for image in cocoEval.dataset["images"]:
    #     # for img_id in coco_api.getImgIds():
    #     basename = image["file_name"]
    #     img_id = image["id"]
    #     cocoEval.params.imgIds = [img_id]
    #     cocoEval.evaluate()
    #     cocoEval.accumulate()
    #     cocoEval.summarize()

    #     # 获取每张图像的 mAP 值
    #     per_image_mAPs.append(cocoEval.stats[:2])
    #     basenames.append(basename)

    # map_pd = pd.DataFrame(per_image_mAPs, index=basenames)

    # # assert save_path.endswith(".json"), " save_path has to end up wiht .json"
    # # with open(save_path, "w") as json_file:
    # #     json.dump(results, json_file)
    return overall_map, map_pd


def evaluate_seg(true_dir, pred_result_dir):
    jsons = glob.glob(f"{pred_result_dir}/*.json")
    metrics = []
    basenames = []
    for json_ in jsons:
        basename = os.path.basename(json_)
        basenames.append(basename)
        with open(json_, "r") as f:
            result = json.load(f)

        # pred_masks = []
        mask = result.pop("masks")
        mask = mask_utils.decode(mask)
        pred_masks = np.transpose(mask, (2, 0, 1))

        true_path = f"{true_dir}/{json_.split('/')[-1].replace('json', 'npy')}"
        inst = np.load(true_path, allow_pickle=True)
        true_masks = [inst == inst_id for inst_id in np.unique(inst)[1:]]
        # 评估 dice, aj

        metric = eveluate_one_pic_inst(true_masks, pred_masks)
        metrics.append(metric)

    metrics = np.array(metrics)
    # metrics = metrics.mean(axis=0)
    metrics = pd.DataFrame(metrics, columns=["dice", "aji"], index=basenames)
    # metrics.to_csv(f"{pred_result_dir}/../metrics.csv")
    avg_metric = np.mean(metrics, axis=0)
    print(metrics.mean(axis=0))
    print(metrics)
    return avg_metric, metrics


# def calculate_map(ann_file, result_json):
#     test = COCO(ann_file)
#     tt = test.loadRes(result_json)
#     cocoEval = COCOeval(test, tt, iouType="segm")
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()


# def evaluate(ann_file, pred_result_dir, true_dir, save_path=None):
#     overall_map, map_pd = evaluate_map(ann_file, pred_result_dir)
#     avg_metric, metrics = evalute_seg(true_dir, pred_result_dir)
#     res = pd.merge(metrics, map_pd, left_index=True, right_index=True)
#     res["average_dice"] = avg_metric[0]
#     res["average_aji"] = avg_metric[1]
#     res["average_map"] = overall_map[0]
#     res["average_map50"] = overall_map[1]
#     if save_path is not None:
#         res.to_csv(save_path)
#     return res


if __name__ == "__main__":
    true_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/inst"
    pred_result_dir = (
        "/root/autodl-tmp/pannuke_app/projects/pannuke/maskrcnn/predict/pred_data/preds"
    )
    ann_file = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/test_annotations.json"
    res = evaluate(evaluate_seg, evaluate_map, ann_file, true_dir, pred_result_dir)
    print("xxx")
