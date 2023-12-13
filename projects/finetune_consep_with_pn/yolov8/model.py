from ultralytics import YOLO

model = YOLO("/root/autodl-tmp/pannuke_app/train/ultralytics/runs/segment/train5/weights/best.pt")
model.train(data="consep.yaml", epochs=100, batch=2, imgsz=1000, freeze=22)
print("x")