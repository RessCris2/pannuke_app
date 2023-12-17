import os

from mmengine import Config
from mmengine.runner import Runner
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class CoNSePDataset(BaseSegDataset):
    classes = ("Epithelial", "Lymphocyte", "Neutrophil", "Macrophage")
    palette = [
        [78, 89, 101],
        [120, 69, 125],
        [53, 125, 34],
        [0, 11, 123],
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
    cfg.train_dataloader.batch_size = 32
    # Set up working dir to save files and logs.
    cfg.work_dir = "work-dir"
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()
