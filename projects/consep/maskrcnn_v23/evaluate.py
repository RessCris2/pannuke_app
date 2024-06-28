import os

import torch
from data_process import CoNSePDataset, get_transform
from detection.engine import evaluate, train_one_epoch
from detection.utils import collate_fn
from model import get_model_instance_segmentation

data_dir="/root/autodl-tmp/pannuke_app/projects/consep/maskrcnn_v23/training_data/"
dataset_val = CoNSePDataset(data_dir=f"{data_dir}/test/256x256_80x80", transforms=get_transform(train=False))


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
val_dl = torch.utils.data.DataLoader(
    dataset_val, batch_size=8, shuffle=False, num_workers=8, pin_memory=True,collate_fn=collate_fn)

model = get_model_instance_segmentation(num_classes=5)
model.load_state_dict(torch.load('epoch6.pth'))
model.eval().to(device)
evaluate(model, val_dl, device='cuda')
print("xxx")
    