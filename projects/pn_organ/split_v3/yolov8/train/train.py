from ultralytics import YOLO

model = YOLO(
    "/root/autodl-tmp/pannuke_app/projects/yolov8n-seg.pt"
)  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="split_v3.yaml", epochs=100, imgsz=640)
print("x")
