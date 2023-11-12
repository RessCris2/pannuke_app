# pannuke_app

[![](https://img.shields.io/badge/blog-@csdn-red.svg)](https://blog.csdn.net/weixin_41783424?spm=1011.2124.3001.5343)

![](https://img.shields.io/github/forks/RessCris2/pannuke_app?style=social)
![](https://img.shields.io/github/watchers/RessCris2/pannuke_app?style=social )
![](https://img.shields.io/github/stars/RessCris2/pannuke_app?color=green&style=social)


Code issues records(To be fixed):
- 数据的输入部分，各种格式的转换有点混乱，而且很容易出错，不容易校验
- dist net 的 vgg 部分是否可以和 unet+ws 部分进行合并？
    - 使用 mmsegmentation 的方式合并完成
- UNet 的训练部分只用到了语义分割的label， 而后续经过后处理评估却用了实例分割的效果。。。？？？？
    觉得可以直接把 ws 处理后的效果和实例分割的效果计算损失来更新模型
- 数据集的描述可以参考 COCO 论文, PASCAL VOC 介绍的信息等。


src:
- data_process 数据转换代码
    - 数据转换过程以 coco 格式 作为核心中枢。每个数据集首先通过一些初始变换，然后输入给 data_transformer 能够提取数据，convert2coco 基于 data_transformer 做上层调用，生成 COCO 格式
    - 在具体处理模型时，针对单个模型可能需要做特定的处理
        - yolo，需要运行代码将 coco格式转换为 yolo 格式
        - mmsegmentation unet，需要在原数据集上做处理，生成 png 格式的label
        - mmsegmentation unet_dist, 需要在原 png 格式label的基础上做 距离变换

    
    在最原始的数据集格式下，生成一份脚本，将数据集处理为结果格式，这个处理过程可以在一台设备上运行模型前快速完成。而不需要保存？或者中间有大数据量生成也不保存？
    还是直接生成结果数据保存为好，但是脚本也要有。

- models: 模型核心代码
    - MaskRCNN
    - HoVerNet
    - UNet
    - UNet-dist
    - YOLO

- evaluation: 评估相关


