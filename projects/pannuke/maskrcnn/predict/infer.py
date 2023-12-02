from mmdet.apis import DetInferencer


def infer_dir(cfg_path, ckpt_path, pred_dir, device="cuda"):
    # 初始化模型
    inferencer = DetInferencer(
        model=cfg_path,
        weights=ckpt_path,
        device=device,
    )
    # 推理示例图片
    inferencer(
        # "/home/pannuke_app/train/datasets/CoNSeP/Train/Images/train_1.png",
        pred_dir,
        batch_size=4,
        pred_score_thr=0.05,
        draw_pred=False,
        out_dir="./pred_data",
        no_save_pred=False,
    )


if __name__ == "__main__":
    root_dir = "/root/autodl-tmp/pannuke_app"
    cfg_path = "../train/model_data/consep_config.py"
    ckpt_path = "../train/model_data/epoch_1.pth"
    pred_dir = f"{root_dir}/datasets/processed/CoNSeP/test/imgs"
    infer_dir(cfg_path, ckpt_path, pred_dir)
