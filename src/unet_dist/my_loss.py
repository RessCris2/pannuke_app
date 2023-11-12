import torch
import torch.nn as nn

from mmseg.registry import MODELS
from mmseg.models.losses.utils import weighted_loss


@weighted_loss
def my_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss


@MODELS.register_module()
class MyLoss(nn.Module):
    def __init__(self, reduction="mean", use_sigmoid=False, loss_weight=1.0):
        """
        use_sigmoid 只是为了统一接口，没有意义
        """
        super(MyLoss, self).__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * my_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss
