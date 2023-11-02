# pannuke_app

[![](https://img.shields.io/badge/blog-@champyin-red.svg)](https://champyin.com)

Code issues records(To be fixed):
- 数据的输入部分，各种格式的转换有点混乱，而且很容易出错，不容易校验
- dist net 的 vgg 部分是否可以和 unet+ws 部分进行合并？
- UNet 的训练部分只用到了语义分割的label， 而后续经过后处理评估却用了实例分割的效果。。。？？？？
    觉得可以直接把 ws 处理后的效果和实例分割的效果计算损失来更新模型
