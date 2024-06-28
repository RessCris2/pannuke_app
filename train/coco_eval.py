from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

# Paths to the files 获取指定图片的数据
gt_json_path = '/root/autodl-tmp/pannuke_app/train/datasets/CoNSeP/Train/annotations.json'  # Ground truth annotations
# Initialize COCO ground truth API
coco_gt = COCO(gt_json_path)
# Initialize COCO detections API
coco_dt = coco_gt.loadRes(detections_json_path)

detections_json_path = 'path/to/your/detections.json'  # Detection results from mmdetection

# Initialize COCO ground truth API
coco_gt = COCO(gt_json_path)

# Initialize COCO detections API
coco_dt = coco_gt.loadRes(detections_json_path)

# COCO Evaluator
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

# Evaluate on a subset of the data, e.g., the validation set. If you want to evaluate all the data, just do not specify the ids.
# coco_eval.params.imgIds = validation_img_ids # Use the ids of your validation images here

# Run Evaluation
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# The results will be printed out, and you can analyze them.