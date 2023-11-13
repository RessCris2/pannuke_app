import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

test = COCO(
    "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/test_annotations.json"
)
pred = test.loadRes("/root/autodl-tmp/pannuke_app/evaluate/test_pred.json")

metric = COCOeval(cocoGt=test, cocoDt=pred, iouType="bbox")
metric.evaluate()
