from ultralytics import YOLO

model = YOLO("pretrained_pannuke.pt")
# load a pretrained model (recommended for training)
# Train the model
results = model.train(data="consep_finetune.yaml", epochs=100, imgsz=640, freeze=18)
print("x")
