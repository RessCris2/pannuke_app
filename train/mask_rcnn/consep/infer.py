from mmdet.apis import DetInferencer

# 初始化模型
inferencer = DetInferencer(
    model="/home/pannuke_app/train/mask_rcnn/consep/consep_config.py",
    weights="./epoch_80.pth",
    device="cpu",
)
# 推理示例图片
inferencer(
    "/home/pannuke_app/train/datasets/CoNSeP/Train/Images/train_1.png",
    show=True,
    pred_score_thr=0.05,
    draw_pred=True,
    out_dir="./",
)
