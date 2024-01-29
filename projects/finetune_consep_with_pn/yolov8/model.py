from ultralytics import YOLO

model = YOLO(
    "/root/autodl-tmp/pannuke_app/train/ultralytics/runs/segment/train5/weights/best.pt"
)
model.train(data="train/consep_finetune.yaml", epochs=30, batch=4, imgsz=640)
print("x")
