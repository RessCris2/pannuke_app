"""测试后如何没有问题，就放进 src 中.
"""
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def calculate_map(ann_file, result_json):
    test = COCO(ann_file)
    tt = test.loadRes(result_json)
    cocoEval = COCOeval(test, tt, iouType="segm")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == "__main__":
    # calculate_map()
    # calculate_pq()
    ann_file = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/test_annotations.json"
    save_path = "/root/autodl-tmp/pannuke_app/projects/consep/maskrcnn/predict/pred_data/preds.json"
    calculate_map(ann_file, save_path)
