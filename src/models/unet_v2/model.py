import torchvision
from torchvision.models.resnet import ResNet50_Weights


class UNet_WS(nn.Module):
    def __init__(self, cfg):
        super(UNet_WS, self).__init__()
        self.cfg = cfg
        self.unet = UNet(
            in_channels=cfg.in_channels,
            n_classes=cfg.n_classes,
            depth=cfg.depth,
            wf=cfg.wf,
            padding=cfg.padding,
            batch_norm=cfg.batch_norm,
            up_mode=cfg.up_mode,
        )

    def forward():
        prob_map = unet(x)
        inst_map = post_proc(prob_map)

        # 处理后的 x 为 inst_map?
        # 要对 inst map 进行后处理，得到 inst_map 和 label_map
        # 可以考虑 先用 dice loss 处理。
        # 之后的 loss 要计算 分类损失和 回归损失啊啊啊啊啊啊啊啊啊啊啊啊啊啊
        return x
