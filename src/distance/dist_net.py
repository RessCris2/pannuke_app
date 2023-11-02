"""
  复现 DIST 算法;
  本质仅仅只是 UNet 模型输出 dist map, 然后按照介绍的 post process 方法处理即可。
  https://zhuanlan.zhihu.com/p/399800147 Pytorch torch.nn库以及nn与nn.functional有什么区别?

  这里的 vgg 部分可能要自己加载数据到模型里。
"""
import torch
from torch import nn
from .nets.unet import Unet


class DIST(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.model = Unet(num_classes=num_features, pretrained=True, backbone="vgg")
        self.conv = nn.Conv2d(in_channels=num_features, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        output = self.conv(x)
        return output


def loss_fn(y_pred, y_true):
    """
    The output result of the DIST model is (B, 1, H, W)
    loss: mean square error
    """
    y_pred = torch.squeeze(y_pred)
    assert y_pred.shape == y_true.shape, "loss compute, dim must be same."
    loss = nn.functional.mse_loss(y_pred, y_true)
    return loss


if __name__ == "__main__":
    x = torch.rand(size=(4, 3, 256, 256))
    model = DIST(num_features=6)
    y = model(x)
    print("xxx")
