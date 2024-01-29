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

# from ultralytics.cfg import get_cfg
# from ultralytics.models.yolo import classify, detect, segment
# from ultralytics.utils import ASSETS, DEFAULT_CFG, WEIGHTS_DIR

# CFG_DET = "yolov8n.yaml"
# CFG = get_cfg(DEFAULT_CFG)

# overrides = {
#     "data": "./consep.yaml",
#     # "data": "./pannuke.yaml",
#     # "data": "./coco8.yaml",
#     "model": "yolov8n.pt",
#     # "model": "yolov5n6u.pt",
#     "imgsz": 256,
#     "epochs": 100,
#     "save": False,
#     # "wandb": None,
#     "batch": 16,
# }
# # CFG.data = "consep.yaml"
# # CFG.imgsz = 1000

# # Trainer
# trainer = detect.DetectionTrainer(overrides=overrides)
# trainer.train()

# model = YOLO("yolov8s-seg.pt")
model = YOLO(
    "/root/autodl-tmp/pannuke_app/train/ultralytics/runs/segment/train18/weights/best.pt"
)
model.train(data="consep.yaml", epochs=100, batch=2, imgsz=800)

# Load a model
# model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
# model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
# # model = YOLO("yolov8n-seg.yaml").load(
# # "yolov8n.pt"
# # )  # build from YAML and transfer weights

# # Train the model
# results = model.train(data="consep.yaml", epochs=500, batch=2, imgsz=1024)
# print("x")


# model = YOLO(
#     "/root/autodl-tmp/pannuke_app/train/ultralytics/runs/segment/train9/weights/last.pt"
# )
# model.train(resume=True)
