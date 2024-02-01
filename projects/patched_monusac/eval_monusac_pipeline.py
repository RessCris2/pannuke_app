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
        "/root/autodl-tmp/pannuke_app/projects/monusac/hovernet/predict/pred_data/mat"
    )
    true_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/test/inst"
    ann_file = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/test/test_annotations.json"
    save_path = (
        "/root/autodl-tmp/pannuke_app/projects/monusac/hovernet/evaluation/hovernet.csv"
    )
    evaluate(evaluate_seg, evaluate_map, ann_file, pred_dir, true_dir, save_path)


def evaluate_hover_patched():
    from src.evaluation.evaluate import evaluate
    from src.evaluation.evaluate_hover import evaluate_map, evaluate_seg

    pred_dir = "/root/autodl-tmp/pannuke_app/projects/patched_monusac/hovernet/predict/pred_data/mat"
    true_dir = "/root/autodl-tmp/pannuke_app/projects/monusac/training_data/test/inst"
    ann_file = "/root/autodl-tmp/pannuke_app/projects/monusac/training_data/test/test_annotations.json"
    save_path = "/root/autodl-tmp/pannuke_app/projects/patched_monusac/hovernet/evaluation/hovernet.csv"
    evaluate(evaluate_seg, evaluate_map, ann_file, pred_dir, true_dir, save_path)


def evaluate_maskrcnn():
    # 需要先运行 infer
    from projects.patched_consep.maskrcnn.predict.infer import infer_dir

    root_dir = "/root/autodl-tmp/pannuke_app"
    cfg_path = "/root/autodl-tmp/pannuke_app/projects/patched_consep/maskrcnn/train/work-dir01/config.py"
    ckpt_path = "/root/autodl-tmp/pannuke_app/projects/patched_consep/maskrcnn/train/work-dir01/epoch_20.pth"
    pred_dir = f"{root_dir}/datasets/processed/CoNSeP/test/imgs"
    infer_dir(cfg_path, ckpt_path, pred_dir)
    from src.evaluation.evaluate import evaluate
    from src.evaluation.evaluate_mr import evaluate_map, evaluate_seg

    true_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/inst"
    pred_dir = "/root/autodl-tmp/pannuke_app/projects/patched_consep/maskrcnn/predict/pred_data/preds"
    ann_file = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/test_annotations.json"
    save_path = "/root/autodl-tmp/pannuke_app/projects/patched_consep/maskrcnn/predict/maskrcnn.csv"
    res = evaluate(evaluate_seg, evaluate_map, ann_file, pred_dir, true_dir, save_path)


def evaluate_maskrcnn_patched():
    # 需要先运行 infer
    from projects.monusac.maskrcnn.predict.infer import infer_dir

    root_dir = "/root/autodl-tmp/pannuke_app"
    cfg_path = "/root/autodl-tmp/pannuke_app/projects/monusac/maskrcnn/train/config.py"
    ckpt_path = (
        "/root/autodl-tmp/pannuke_app/projects/monusac/maskrcnn/train/epoch_7.pth"
    )
    pred_dir = "/root/autodl-tmp/pannuke_app/projects/monusac/training_data/test/imgs"
    out_dir = "/root/autodl-tmp/pannuke_app/projects/monusac/maskrcnn/predict/pred_data"
    infer_dir(cfg_path, ckpt_path, pred_dir, out_dir)
    from src.evaluation.evaluate import evaluate
    from src.evaluation.evaluate_mr import evaluate_map, evaluate_seg

    true_dir = "/root/autodl-tmp/pannuke_app/projects/monusac/training_data/test/inst"
    pred_dir = f"{out_dir}/preds"
    ann_file = "/root/autodl-tmp/pannuke_app/projects/monusac/training_data/test/test_annotations.json"
    save_path = (
        "/root/autodl-tmp/pannuke_app/projects/monusac/maskrcnn/evaluation/maskrcnn.csv"
    )
    res = evaluate(evaluate_seg, evaluate_map, ann_file, pred_dir, true_dir, save_path)


def evaluate_unet_patched():
    """具体看是用training data 还是原始数据"""
    from src.evaluation.evaluate_unet import evaluate_unet

    pred_dir = "/root/autodl-tmp/pannuke_app/projects/monusac/training_data/test/imgs"
    cfg_path = (
        "/root/autodl-tmp/pannuke_app/projects/monusac/unet/train/work-dir/config.py"
    )
    ckpt_filename = "/root/autodl-tmp/pannuke_app/projects/monusac/unet/train/work-dir/iter_40000.pth"
    ann_file = "/root/autodl-tmp/pannuke_app/projects/monusac/training_data/test/test_annotations.json"
    mid_save_dir = "/root/autodl-tmp/pannuke_app/projects/monusac/unet/predict/inst/"
    save_path = "/root/autodl-tmp/pannuke_app/projects/monusac/unet/evaluation/unet.csv"
    res = evaluate_unet(
        cfg_path, ckpt_filename, pred_dir, ann_file, mid_save_dir, save_path
    )
    print(res)


def evaluate_yolo():
    from src.evaluation.evaluate import evaluate
    from src.evaluation.evaluate_yolo import evaluate_metric

    ann_file = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/test/test_annotations.json"

    pred_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/test/imgs"
    best_path = "/root/autodl-tmp/pannuke_app/train/ultralytics/runs/segment/train10/weights/best.pt"

    true_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/test/inst"
    model = YOLO(model=best_path)  # load a custom model
    # true_masks_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/inst"
    save_path = (
        "/root/autodl-tmp/pannuke_app/projects/monusac/yolov8/evaluation/yolov8.csv"
    )
    overall_map, map_pd, avg_metric, metrics = evaluate_metric(
        model, pred_dir, ann_file, true_dir, save_path
    )
    print("result")


def evaluate_yolo_patched():
    from src.evaluation.evaluate import evaluate
    from src.evaluation.evaluate_yolo import evaluate_metric

    ann_file = "/root/autodl-tmp/pannuke_app/projects/monusac/training_data/test/test_annotations.json"

    pred_dir = "/root/autodl-tmp/pannuke_app/projects/monusac/training_data/test/imgs"
    best_path = "/root/autodl-tmp/pannuke_app/train/ultralytics/runs/segment/train21/weights/best.pt"

    true_dir = "/root/autodl-tmp/pannuke_app/projects/monusac/training_data/test/inst"
    model = YOLO(model=best_path)  # load a custom model
    # true_masks_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/test/inst"
    save_path = (
        "/root/autodl-tmp/pannuke_app/projects/monusac/yolov8/evaluation/yolov8.csv"
    )
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
    evaluate_hover_patched()