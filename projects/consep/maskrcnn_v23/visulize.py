import argparse
import time

import cv2
import numpy as np
import torch
from model import get_model_instance_segmentation
from torchvision.models.detection import maskrcnn_resnet50_fpn
from utils import (draw_boxes, draw_instance_mask, filter_detections,
                   get_transformed_image)

# Set NumPy seed.
np.random.seed(2022)

COCO_INSTANCE_CATEGORY_NAMES=[
    '__background__', 'person', 'bicycle', 'car', 'motorcycle'
]
colors=np.random.randint(0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))
colors = [tuple(color) for color in colors]


# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = maskrcnn_resnet50_fpn(pretrained=True)
model = get_model_instance_segmentation(num_classes=5)
model.load_state_dict(torch.load('epoch20.pth'))
model.eval().to(device)

frame = cv2.imread("/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/imgs/test_1.png")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
uint8_frame, float32_frame = get_transformed_image(frame)

with torch.no_grad():
    # Get predictions for the current frame.
    outputs = model(float32_frame.to(device))

# Get the filetered boxes, class names, and label indices.
boxes, pred_classes, labels = filter_detections(
    outputs, COCO_INSTANCE_CATEGORY_NAMES,
   0.05
)

# Draw boxes and show current frame on screen.
result, plot_colors = draw_boxes(
    boxes, uint8_frame, pred_classes, 
    labels, colors, is_instance=True
)

result = draw_instance_mask(outputs, result, plot_colors, 0.05)
result = np.transpose(result, (1, 2, 0))
result = np.ascontiguousarray(result, dtype=np.uint8)
# Convert from BGR to RGB color format.
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
print(result)