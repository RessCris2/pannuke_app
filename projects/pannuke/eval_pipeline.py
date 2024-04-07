"""
    写一个相对完整的评估; 生成每幅图的 DICE, AJI, mAP
"""
# hovernet
import sys

from ultralytics import YOLO

sys.path.append("/root/autodl-tmp/pannuke_app/")


def evaluate_hover():
    from src.evaluation.evaluate import evaluate
    from src.evaluation.evaluate_hover import evaluate_map, evaluate_seg

    pred_dir = (
        "/root/autodl-tmp/pannuke_app/projects/pannuke/hovernet/predict/pred_data/mat"
    )
    true_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/inst"
    ann_file = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/test_annotations.json"
    save_path = (
        "/root/autodl-tmp/pannuke_app/projects/pannuke/hovernet/evaluation/hovernet.csv"
    )

    evaluate(evaluate_seg, evaluate_map, ann_file, pred_dir, true_dir, save_path)


def evaluate_maskrcnn():
    from src.evaluation.evaluate import evaluate
    from src.evaluation.evaluate_mr import evaluate_map, evaluate_seg

    pred_dir = (
        "/root/autodl-tmp/pannuke_app/projects/pannuke/hovernet/predict/pred_data/mat"
    )
    true_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/inst"
    ann_file = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/test_annotations.json"
    save_path = (
        "/root/autodl-tmp/pannuke_app/projects/pannuke/maskrcnn/evaluation/maskrcnn.csv"
    )

    evaluate(evaluate_seg, evaluate_map, ann_file, pred_dir, true_dir, save_path)


def evaluate_unet():
    from src.evaluation.evaluate import evaluate
    from src.evaluation.evaluate_unet import evaluate_map, evaluate_seg

    pred_dir = (
        "/root/autodl-tmp/pannuke_app/projects/pannuke/hovernet/predict/pred_data/mat"
    )
    true_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/inst"
    ann_file = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/test_annotations.json"
    save_path = "/root/autodl-tmp/pannuke_app/projects/pannuke/unet/evaluation/unet.csv"
    best_path = "/root/autodl-tmp/pannuke_app/train/ultralytics/runs/segment/train5/weights/best.pt"
    model = YOLO(model=best_path)

    evaluate(
        evaluate_seg, evaluate_map, ann_file, pred_dir, true_dir, save_path, model=model
    )


def evaluate_yolo():
    from src.evaluation.evaluate import evaluate
    from src.evaluation.evaluate_yolo import evaluate_metric

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


if __name__ == "__main__":
    # evaluate_maskrcnn()
    # evaluate_yolo()
    # evaluate_unet()
    evaluate_hover()
