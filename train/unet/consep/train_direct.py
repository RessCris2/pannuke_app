import os

from mmengine import Config
from mmengine.runner import Runner
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class CoNSePDataset(BaseSegDataset):
    classes = ("Inflammatory", "Healthy_epithelial", "Epithelial", "Spindle-shaped")
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
    if os.path.exists(""):
        config_path = "/home/pannuke_app/train/unet/consep/config_consep.py"
    else:
        config_path = "/root/autodl-tmp/pannuke_app/train/unet/consep/config_consep.py"
    cfg = Config.fromfile(config_path)
    print(f"Config:\n{cfg.pretty_text}")
    # Modify dataset type and path
    # cfg.dataset_type = "PanNukeDataset"
    # cfg.data_root = "/root/autodl-tmp/datasets/pannuke/coco_format/"
    cfg.train_dataloader.batch_size = 4
    # Set up working dir to save files and logs.
    cfg.work_dir = "/root/autodl-tmp/pannuke_app/train/unet/consep/work-dir"
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()
