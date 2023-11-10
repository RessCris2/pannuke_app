# from ultralytics import YOLO

# model = YOLO(
#     data="consep.yaml",
#     model="yolov8n.pt",
#     imgsz=1000,
#     epochs=10,
#     save=True,
# )


# model.train()
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo import classify, detect, segment
from ultralytics.utils import ASSETS, DEFAULT_CFG, WEIGHTS_DIR

# CFG_DET = "yolov8n.yaml"
# CFG = get_cfg(DEFAULT_CFG)

overrides = {
    # "data": "./consep.yaml",
    "data": "./pannuke.yaml",
    # "data": "./coco8.yaml",
    # "model": "yolov8n.pt",
    "model": "yolov5n6u.pt",
    "imgsz": 256,
    "epochs": 100,
    "save": False,
    # "wandb": None,
    "batch": 16,
}
# CFG.data = "consep.yaml"
# CFG.imgsz = 1000

# Trainer
trainer = detect.DetectionTrainer(overrides=overrides)
trainer.train()
