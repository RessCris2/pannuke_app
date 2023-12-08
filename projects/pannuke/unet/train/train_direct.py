import os

from mmengine import Config
from mmengine.runner import Runner
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class PanNukeDataset(BaseSegDataset):
    classes = ("Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial")
    palette = [
        (200, 10, 60),
        (120, 120, 60),
        (20, 120, 160),
        (72, 100, 60),
        (111, 67, 60),
    ]
    METAINFO = dict(classes=classes, palette=palette)

    def __init__(self, **kwargs):
        super().__init__(img_suffix=".png", seg_map_suffix=".png", **kwargs)


if __name__ == "__main__":
    config_path = "config.py"
    cfg = Config.fromfile(config_path)
    print(f"Config:\n{cfg.pretty_text}")
    # Modify dataset type and path
    # cfg.dataset_type = "PanNukeDataset"
    # cfg.data_root = "/root/autodl-tmp/datasets/pannuke/coco_format/"
    # cfg.train_dataloader.batch_size = 64
    # cfg.val_dataloader.batch_size = 64
    # Set up working dir to save files and logs.
    cfg.work_dir = "work-dir"
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()
