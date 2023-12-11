import os

import torch
import torchvision
from data_process import CoNSePDataset, get_transform
from detection.engine import evaluate, train_one_epoch
from detection.utils import collate_fn
from model import get_model_instance_segmentation
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./logs/02/')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EPOCHS = 30

data_dir="/root/autodl-tmp/pannuke_app/projects/consep/maskrcnn_v23/training_data/"
dataset_train = CoNSePDataset(data_dir=f"{data_dir}/train/256x256_80x80", transforms= get_transform(train=True))
dataset_val = CoNSePDataset(data_dir=f"{data_dir}/valid/256x256_80x80", transforms=get_transform(train=False))

train_dl = torch.utils.data.DataLoader(
        dataset_train, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)
val_dl = torch.utils.data.DataLoader(
    dataset_val, batch_size=8, shuffle=False, num_workers=8, pin_memory=True,collate_fn=collate_fn)

model = get_model_instance_segmentation(num_classes=5,)
model.load_state_dict(torch.load('epoch7.pth'))
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.0003, weight_decay=1e-6)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

for epoch in range(8, EPOCHS):
    metric = train_one_epoch(model, optimizer, train_dl, device, epoch, print_freq=1)
    # scheduler.step()
    writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch)
    writer.add_scalar('loss_classifier', metric.meters['loss_classifier'].avg, epoch)
    writer.add_scalar('loss_box_reg', metric.meters['loss_box_reg'].avg, epoch)
    writer.add_scalar('loss_mask', metric.meters['loss_mask'].avg, epoch)
    writer.add_scalar('loss_objectness', metric.meters['loss_objectness'].avg, epoch)
    writer.add_scalar('loss_rpn_box_reg', metric.meters['loss_rpn_box_reg'].avg, epoch)
    writer.add_scalar('loss', metric.meters['loss'].avg, epoch)
    writer.add_scalar('epoch', epoch, epoch)

    if (epoch % 5 == 0 and epoch != 0) or epoch == EPOCHS - 1:
        model_path = f'epoch{epoch}.pth'
        torch.save(model.state_dict(), model_path)

    if (epoch % 10 == 0 and epoch != 0) or epoch == EPOCHS - 1:
        evaluate(model, val_dl, device=device)
        model_path = f'epoch{epoch}.pth'
        torch.save(model.state_dict(), model_path)
        model = model.to(device)

# grid = torchvision.utils.make_grid(images)
# writer.add_image('images', grid, 0)
# writer.add_graph(model, images)
writer.close()


# 使用 tensorboard 记录损失和梯度图？
# 训练时使用 预热？
# resume 是怎么做的？
# evaluate 报错