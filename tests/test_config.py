"""Valiate the config read from file.
配置类提供了统一的接口 Config.fromfile()，来读取和解析配置文件。

提供了两种访问接口，即类似字典的接口 cfg['key'] 或者类似 Python 对象属性的接口 cfg.key。这两种接口都支持读写。

进阶用法在这 https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/config.html
"""

from mmengine.config import Config

# cfg = Config.fromfile("learn_read_config.py")
# print(cfg)

# # dump the cfg file
# cfg = Config.fromfile("resnet50.py")
# cfg.dump("resnet50_dump.py")

# DictAction 的行为与 "extend" 相似，支持多次传递，并保存在同一个列表中


# 纯Python风格的配置文件
# 现有的配置文件不支持跳转，很烦
## 模块构建、继承和导出

# from torch.optim import SGD

# optimizer = dict(type=SGD, lr=0.1)

# # 构建流程完全一致
# import torch.nn as nn
# from mmengine.registry import OPTIMIZERS


# cfg = Config.fromfile("optimizer.py")
# model = nn.Conv2d(1, 1, 1)
# cfg.optimizer.params = model.parameters()
# optimizer = OPTIMIZERS.build(cfg.optimizer)


# Reduce the number of repeated compilations and improve
# testing speed.
# load config
cfg = Config.fromfile("/home/pannuke_app/train/mask_rcnn/consep/consep_config.py")
print(cfg)
# cfg.launcher = args.launcher
# if args.cfg_options is not None:
#     cfg.merge_from_dict(args.cfg_options)
