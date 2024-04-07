"""
    写一个相对完整的评估; 生成每幅图的 DICE, AJI, mAP
"""
# hovernet
import sys

from ultralytics import YOLO

sys.path.append("/root/autodl-tmp/pannuke_app/")


def evaluate_yolo_v1():
    from src.evaluation.evaluate import evaluate
    from src.evaluation.evaluate_yolo import evaluate_metric

    ann_file = "/root/autodl-tmp/pannuke_app/projects/pn_organ/split_v1/training_data/test/test_annotations.json"

    pred_dir = "/root/autodl-tmp/pannuke_app/projects/pn_organ/split_v1/training_data/test/imgs"
    best_path = "/root/autodl-tmp/pannuke_app/train/ultralytics/runs/segment/split_v1/weights/best.pt"

    true_dir = "/root/autodl-tmp/pannuke_app/projects/pn_organ/split_v1/training_data/test/inst"
    model = YOLO(model=best_path)  # load a custom model
    save_path = "/root/autodl-tmp/pannuke_app/projects/pn_organ/split_v1/yolov8/evaluation/yolov8.csv"
    overall_map, map_pd, avg_metric, metrics = evaluate_metric(
        model, pred_dir, ann_file, true_dir, save_path
    )
    print("result")


def evaluate_yolo_v2():
    from src.evaluation.evaluate import evaluate
    from src.evaluation.evaluate_yolo import evaluate_metric

    ann_file = "/root/autodl-tmp/pannuke_app/projects/pn_organ/split_v2/training_data/test/test_annotations.json"

    pred_dir = "/root/autodl-tmp/pannuke_app/projects/pn_organ/split_v2/training_data/test/imgs"
    best_path = "/root/autodl-tmp/pannuke_app/train/ultralytics/runs/segment/split_v2/weights/best.pt"

    true_dir = "/root/autodl-tmp/pannuke_app/projects/pn_organ/split_v2/training_data/test/inst"
    model = YOLO(model=best_path)  # load a custom model
    save_path = "/root/autodl-tmp/pannuke_app/projects/pn_organ/split_v2/yolov8/evaluation/yolov8.csv"
    overall_map, map_pd, avg_metric, metrics = evaluate_metric(
        model, pred_dir, ann_file, true_dir, save_path
    )
    print("result")


def evaluate_yolo_v3():
    from src.evaluation.evaluate import evaluate
    from src.evaluation.evaluate_yolo import evaluate_metric

    ann_file = "/root/autodl-tmp/pannuke_app/projects/pn_organ/split_v3/training_data/test/test_annotations.json"

    pred_dir = "/root/autodl-tmp/pannuke_app/projects/pn_organ/split_v3/training_data/test/imgs"
    best_path = "/root/autodl-tmp/pannuke_app/train/ultralytics/runs/segment/split_v3/weights/best.pt"

    true_dir = "/root/autodl-tmp/pannuke_app/projects/pn_organ/split_v3/training_data/test/inst"
    model = YOLO(model=best_path)  # load a custom model
    save_path = "/root/autodl-tmp/pannuke_app/projects/pn_organ/split_v3/yolov8/evaluation/yolov8.csv"
    overall_map, map_pd, avg_metric, metrics = evaluate_metric(
        model, pred_dir, ann_file, true_dir, save_path
    )
    print("result")


if __name__ == "__main__":
    # evaluate_maskrcnn()
    # evaluate_maskrcnn_patched()
    # evaluate_yolo()
    # evaluate_yolo_patched()
    # evaluate_unet()
    # evaluate_hover()  # 非常慢，而且会莫名中断。考虑用 patched 的数据集来做
    # evaluate_hover_patched()
    evaluate_yolo_v1()
    evaluate_yolo_v2()
    evaluate_yolo_v3()
