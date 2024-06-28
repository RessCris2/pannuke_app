"""熟悉 coco metric
"""
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# coco = COCO(
#     annotation_file="/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/test_annotations.json"
# )

# coco = COCO()
"""为当前的 images 添加 ann"""
# coco.loadRes("/root/autodl-tmp/pannuke_app/predict/maskrcnn/preds/test_1.json")

test = COCO(
    "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/test_annotations.json"
)
results = []
for image in test.dataset["images"]:
    file_name = "/root/autodl-tmp/pannuke_app/predict/maskrcnn/preds/{}".format(
        image["file_name"].replace("png", "json")
    )
    with open(file_name, "r") as f:
        result = json.load(f)
    result["image_id"] = image["id"]
    result["bbox"] = result.pop("bboxes")  # bbox 还得改为 xyxy 改为 xywh 格式
    result["category_id"] = result.pop("labels")
    result["segmentation"] = result.pop("masks")
    result["score"] = result.pop("scores")

    # 将 result 中的列表格式改为 字典格式，方便 coco.loadRes
    # trans = []
    items = []

    for idx, (d1, d2, d3, d4) in enumerate(
        zip(
            result["bbox"],
            result["category_id"],
            result["segmentation"],
            result["score"],
        )
    ):
        item = {}
        # print(d1, d2, d3, d4  )
        # item.update({"id": idx + 1})
        item.update({"bbox": d1})
        item.update({"category_id": d2})
        item.update({"segmentation": d3})
        item.update({"score": d4})
        item.update({"image_id": image["id"]})
        items.append(item)

    results.extend(items)


with open("/root/autodl-tmp/pannuke_app/evaluate/test_pred.json", "w") as json_file:
    json.dump(results, json_file)

# tt = test.loadRes(results)


"""

annotations:
    - image_id
    - category_id
    - bbox
    - segmentation(optional)
    - score(optional)

    
    # now only support compressed RLE format as segmentation results 
    这一步没看懂，好像没处理 segmentation 的部分？？？
    # 如果包含 bbox， 就不会处理 segmentation

    # 另外 score？
    scores 会自动称为 ann 
"""
