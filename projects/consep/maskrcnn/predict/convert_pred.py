""" 将 maskrcnn 的预测结果转换为 coco 格式，方便后续的评估
"""
import json

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def convert_pred2coco(ann_file, pred_result_dir):
    # test = COCO(
    #     "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/test_annotations.json"
    # )
    test = COCO(ann_file)
    results = []
    for image in test.dataset["images"]:
        file_name = "{}/{}".format(
            pred_result_dir, image["file_name"].replace("png", "json")
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


def convert_pred2masks():
    """将数据集转换为适用于计算 pq, dice, aji+ 的格式"""
    pass


# # 假设你有一个 mmdetection 输出的 mask
# mask = np.random.randint(2, size=(100, 100), dtype=np.uint8)

# # 将二进制掩码转换为 RLE 编码
# rle_encoded = mask_utils.encode(np.array(mask[:, :, None], order="F"))[0]

# # 将 RLE 编码解码为二进制掩码
# decoded_mask = mask_utils.decode([rle_encoded])

# # 显示原始 mask 和解码后的 mask
# mmcv.imshow_bboxes(
#     "Original Mask", [mask], colors=[(0, 255, 0)], thickness=2, show=True
# )
# mmcv.imshow_bboxes(
#     "Decoded Mask", [decoded_mask], colors=[(0, 255, 0)], thickness=2, show=True
# )


if __name__ == "__main__":
    ann_file = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/test_annotations.json"
    pred_result_dir = "/root/autodl-tmp/pannuke_app/predict/pred_data"
    convert_pred2coco(ann_file, pred_result_dir)
