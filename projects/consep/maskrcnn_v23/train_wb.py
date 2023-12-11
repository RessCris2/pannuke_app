import os

import torch
from data_process import CoNSePDataset, get_transform
from detection.engine import evaluate, train_one_epoch
from detection.utils import collate_fn
from model import get_model_instance_segmentation
from torch.utils.tensorboard import SummaryWriter
import torchvision
import wandb
from argparse import Namespace

config = Namespace()
wandb.init(project="consep_maskrcnn", entity="consep")

sweep_config = {
    "method": 'random', #grid, random

}

metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric
sweep_config['parameters'] = {}

# 固定不变的超参
sweep_config['parameters'].update({
    'project_name':{'value':'wandb_demo'},
    'epochs': {'value': 10},
    'ckpt_path': {'value':'checkpoint.pt'}})

# 离散型分布超参
sweep_config['parameters'].update({
    'optim_type': {
        'values': ['Adam', 'SGD','AdamW']
        },
    'hidden_layer_width': {
        'values': [16,32,48,64,80,96,112,128]
        }
    })

# 连续型分布超参
sweep_config['parameters'].update({
    
    'lr': {
        'distribution': 'log_uniform_values',
        'min': 1e-6,
        'max': 0.1
      },
    
    'batch_size': {
        'distribution': 'q_uniform',
        'q': 8,
        'min': 32,
        'max': 256,
      },
    
    'dropout_p': {
        'distribution': 'uniform',
        'min': 0,
        'max': 0.6,
      }
})


sweep_id = wandb.sweep(sweep_config, project=config.project_name)

def train(config):
    pass

# 该agent 随机搜索 尝试5次
wandb.agent(sweep_id, train, count=5)

writer = SummaryWriter()



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EPOCHS = 500

dataset_train = CoNSePDataset(data_dir="/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/train/", transforms= get_transform(train=True))
dataset_val = CoNSePDataset(data_dir="/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/", transforms=get_transform(train=False))

train_dl = torch.utils.data.DataLoader(
        dataset_train, batch_size=8, shuffle=True, num_workers=os.cpu_count(), pin_memory=True, drop_last=True, collate_fn=collate_fn)
val_dl = torch.utils.data.DataLoader(
    dataset_val, batch_size=8, shuffle=False, num_workers=os.cpu_count(), pin_memory=True,collate_fn=collate_fn)

model = get_model_instance_segmentation(num_classes=4)
model.load_state_dict(torch.load('epoch9.pth'))
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.0003, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

for epoch in range(EPOCHS):
    metric = train_one_epoch(model, optimizer, train_dl, device, epoch, print_freq=1)
    scheduler.step()

    writer.add_scalars(f'epoch_{epoch}',{'loss': metric.meters['loss'].avg, 
                         'lr': optimizer.param_groups[0]["lr"],
                         'loss_classifier': metric.meters['loss_classifier'].avg,
                         'loss_box_reg': metric.meters['loss_box_reg'].avg,
                         'loss_mask': metric.meters['loss_mask'].avg,
                         'loss_objectness': metric.meters['loss_objectness'].avg,
                         'loss_rpn_box_reg': metric.meters['loss_rpn_box_reg'].avg,
                         'loss': metric.meters['loss'].avg,
                         'epoch': epoch,
                         })

   

    if (epoch % 10 == 0 and epoch != 0) or epoch == EPOCHS - 1:
        evaluate(model.cpu(), val_dl, device='cpu')
        model_path = f'epoch{epoch}.pth'
        torch.save(model.state_dict(), model_path)



# grid = torchvision.utils.make_grid(images)
# writer.add_image('images', grid, 0)
# writer.add_graph(model, images)
writer.close()


# 使用 tensorboard 记录损失和梯度图？
# 训练时使用 预热？
# resume 是怎么做的？
# evaluate 报错