from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

# Train the model
results = model.train(data="pannuke.yaml", epochs=100, imgsz=640)
