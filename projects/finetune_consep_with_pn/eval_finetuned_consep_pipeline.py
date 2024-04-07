"""
    写一个相对完整的评估; 生成每幅图的 DICE, AJI, mAP
"""
# hovernet
import sys

from ultralytics import YOLO

sys.path.append("/root/autodl-tmp/pannuke_app/")



def evaluate_yolo():
    from src.evaluation.evaluate import evaluate
    from src.evaluation.evaluate_yolo import evaluate_metric

    ann_file = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/test_annotations.json"

    pred_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/imgs"
    best_path = "/root/autodl-tmp/pannuke_app/train/ultralytics/runs/segment/train20/weights/best.pt"

    true_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/inst"
    model = YOLO(model=best_path)  # load a custom model
    # true_masks_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/inst"
    save_path = "/root/autodl-tmp/pannuke_app/projects/patched_consep/yolov8/evaluation/yolov8.csv"
    overall_map, map_pd, avg_metric, metrics = evaluate_metric(
        model, pred_dir, ann_file, true_dir, save_path
    )
    print("result")


def evaluate_yolo_patched():
    from src.evaluation.evaluate import evaluate
    from src.evaluation.evaluate_yolo import evaluate_metric

    ann_file = "/root/autodl-tmp/pannuke_app/projects/patched_consep/training_data/test/test_annotations.json"

    pred_dir = (
        "/root/autodl-tmp/pannuke_app/projects/patched_consep/training_data/test/imgs"
    )
#     best_path = "/root/autodl-tmp/pannuke_app/train/ultralytics/runs/segment/train23/weights/best.pt"
    best_path = "/root/autodl-tmp/pannuke_app/projects/finetune_consep_with_pn/segment/train5/weights/best.pt"

    true_dir = (
        "/root/autodl-tmp/pannuke_app/projects/patched_consep/training_data/test/inst"
    )
    model = YOLO(model=best_path)  # load a custom model
    # true_masks_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/inst"
    save_path = "/root/autodl-tmp/pannuke_app/projects/finetune_consep_with_pn/yolov8/evaluation/yolov8.csv"
    overall_map, map_pd, avg_metric, metrics = evaluate_metric(
        model, pred_dir, ann_file, true_dir, save_path
    )
    print("result")


if __name__ == "__main__":
    # evaluate_maskrcnn()
    # evaluate_maskrcnn_patched()
    # evaluate_unet_patched()
    evaluate_yolo_patched()
    # evaluate_hover()
