import os
import sys

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

sys.path.append("/root/autodl-tmp/pannuke_app")
import cv2
import numpy as np
from src.data_process.pycococreatortools import binary_mask_to_polygon
from src.evaluation.evaluate import calculate_map
from src.evaluation.stats_utils_v2 import eveluate_one_pic_inst

""" bbox 的格式是不准的，和 mask 不是最贴合的。算不准，不要用。
"""


def polygon_to_binary_mask(polygon, image_shape):
    # 创建一个黑色图像
    mask = np.zeros(image_shape, dtype=np.uint8)
    # 将多边形坐标转换为整数
    polygon = np.array(polygon, dtype=np.int32)
    # 在图像上绘制多边形
    cv2.fillPoly(mask, [polygon], 1)
    # 将图像转换为二进制掩模
    binary_mask = (mask == 1).astype(np.uint8)
    return binary_mask


def find_img_id(file_name, coco_api):
    """根据文件名找到对应的 img_id"""
    image_id = None
    for img_id, img_info in coco_api.imgs.items():
        if img_info["file_name"] == file_name:
            image_id = img_id
            break
    return image_id


# yolo 这个因为是没有提前预测保存数据，还是把 map， dice, aji 一起算。
def evaluate_metric(model, pred_dir, ann_file, true_dir, save_path=None):
    preds = model.predict(
        pred_dir,
        stream=True,
        save=False,
        save_dir="./pred_data",
        retina_masks=True,
        show_labels=True,
    )

    coco_api = COCO(ann_file)
    results = []
    metrics = []
    basenames = []
    for pred in preds:
        try:
            pred = pred.cpu()
        except:
            print("pred is on cpu")
        basename = pred.path.split("/")[-1]
        path = pred.path
        # find_img_id 这个path 要的是 全名还是末尾的名字？
        result = dict()
        if len(pred.boxes) > 0:
            img_id = find_img_id(basename, coco_api)
            # result["image_id"] = find_img_id(basename, coco_api)
            result["bbox"] = pred.boxes.xywh.numpy()
            result["category_id"] = pred.boxes.cls.numpy().tolist()  # 0,1,2,3
            # result["segmentation"] = pred.masks.xy  # 去除masks为0的情况？xy pixels 不行，后面用不了
            result["segmentation"] = pred.masks
            result["score"] = pred.boxes.conf.numpy()
            # results.append(result)

            items = []
            # pred_masks = []
            for d1, d2, d3, d4 in zip(
                result["bbox"],
                result["category_id"],
                result["segmentation"],
                result["score"],
            ):
                item = {}
                item.update({"bbox": d1})
                item.update({"category_id": d2 + 1})

                # binary_mask = polygon_to_binary_mask(
                #     d3.tolist(), image_shape=pred.orig_shape
                # )
                binary_mask = d3.data.numpy().squeeze()
                # binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
                segmmentation = binary_mask_to_polygon(binary_mask)
                item.update({"score": d4})
                item.update({"image_id": img_id})
                item.update({"segmentation": segmmentation})
                items.append(item)

            results.extend(items)
            # basenames.append(basename)

        else:
            print(basename, "has no masks")
            continue

        # basename = path.split("/")[-1].split(".")[0]
        inst_path = f"{true_dir}/{path.split('/')[-1].replace('png', 'npy')}"
        true_inst = np.load(inst_path)
        true_masks = [true_inst == inst_id for inst_id in np.unique(true_inst)[1:]]
        # boxes = result.boxes  # Boxes object for bbox outputs
        masks = pred.masks  # Masks object for segmentation masks outputs
        pred_masks = [mask.data.numpy().squeeze() for mask in masks]
        # keypoints = result.keypoints  # Keypoints object for pose outputs
        # probs = result.probs  # Probs object for classification outputs

        metric = eveluate_one_pic_inst(true_masks, pred_masks)
        print(metric)
        basenames.append(basename)
        metrics.append(metric)

    avg_metric = np.array(metrics).mean(axis=0)
    metrics_pd = pd.DataFrame(metrics, columns=["dice", "aji"], index=basenames)
    # metrics["basename"] = basenames
    print(metrics)
    # return

    overall_map, map_pd = calculate_map(results, coco_api)

    res = pd.merge(metrics_pd, map_pd, left_index=True, right_index=True)
    res["average_dice"] = avg_metric[0]
    res["average_aji"] = avg_metric[1]
    res["average_map"] = overall_map[0]
    res["average_map50"] = overall_map[1]

    if save_path is not None:
        res.to_csv(save_path)

    return overall_map, map_pd, avg_metric, metrics


# def evaluate_seg(true_dir, results):
#     # true_paths = glob.glob(os.path.join(true_dir, "*.npy"))

#     metrics = []
#     basenames = []
#     for true_path in true_paths:
#         basename = pathlib.Path(true_path).stem
#         pred_path = os.path.join(pred_dir, basename + ".mat")
#         try:
#             true_masks = convert_inst2masks(true_path)
#             pred_masks = convert_mat2masks(pred_path)  # 预测的数据里可能会没有instance
#         except ValueError:
#             continue
#         basenames.append(basename)
#         metric = eveluate_one_pic_inst(true_masks, pred_masks)
#         metrics.append(metric)

#     metrics = []
#     basenames = []
#     for result in results:
#         if result.masks is None:
#             continue
#         path = result.path  # img filename
#         basename = path.split("/")[-1].split(".")[0]
#         inst_path = f"{true_dir}/{path.split('/')[-1].replace('png', 'npy')}"
#         true_inst = np.load(inst_path)
#         true_masks = [true_inst == inst_id for inst_id in np.unique(true_inst)[1:]]
#         # boxes = result.boxes  # Boxes object for bbox outputs
#         masks = result.masks  # Masks object for segmentation masks outputs
#         pred_masks = [mask.data.cpu().numpy().squeeze() for mask in masks]
#         # keypoints = result.keypoints  # Keypoints object for pose outputs
#         # probs = result.probs  # Probs object for classification outputs

#         metric = eveluate_one_pic_inst(true_masks, pred_masks)
#         print(metric)
#         basenames.append(basename)
#         metrics.append(metric)

#     avg_metric = np.array(metrics).mean(axis=0)
#     metrics = pd.DataFrame(metrics, columns=["dice", "aji"], index=basenames)
#     # metrics["basename"] = basenames
#     print(metrics)
#     return avg_metric, metrics


if __name__ == "__main__":
    ann_file = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/test_annotations.json"
    # true_masks_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/inst"
    pred_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/imgs"
    best_path = "/root/autodl-tmp/pannuke_app/train/ultralytics/runs/segment/train5/weights/best.pt"
    true_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/inst"
    model = YOLO(model=best_path)  # load a custom model
    # true_masks_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/inst"
    save_path = (
        "/root/autodl-tmp/pannuke_app/projects/pannuke/yolov8/evaluation/yolov8.csv"
    )
    overall_map, map_pd, avg_metric, metrics = evaluate_metric(
        model, pred_dir, ann_file, true_dir, save_path
    )
    print("result")
    # model = YOLO('path/to/best.pt')  # load a custom model
    # model.val()
