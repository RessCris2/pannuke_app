""" 将 maskrcnn 的预测结果转换为 coco 格式，方便后续的评估
"""
import json
import os
import sys

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append("/root/autodl-tmp/pannuke_app")
from src.evaluation.stats_utils_v2 import eveluate_one_pic_inst


def convert_pred2coco(ann_file, pred_result_dir, save_path):
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
    assert save_path.endswith(".json"), " save_path has to end up wiht .json"
    with open(save_path, "w") as json_file:
        json.dump(results, json_file)


"""
annotations:
    - image_id
    - category_id
    - bbox
    - segmentation(optional)
    - score(optional)

    
    # now only support compressed RLE format as segmentation results 
    这一步没看懂，好像没处理 segmentation 的部分？？？
    # 如果包含 bbox, 就不会处理 segmentation

    # 另外 score?
    scores 会自动称为 ann 
"""


if __name__ == "__main__":
    ann_file = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/test_annotations.json"
    pred_result_dir = (
        "/root/autodl-tmp/pannuke_app/projects/pannuke/maskrcnn/predict/pred_data/preds"
    )
    save_path = "/root/autodl-tmp/pannuke_app/projects/pannuke/maskrcnn/predict/pred_data/preds.json"
    convert_pred2coco(ann_file, pred_result_dir, save_path)
    print("x")
