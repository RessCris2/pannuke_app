import torch
import torch.nn as nn
from mmseg.models.losses.utils import weighted_loss
from mmseg.registry import MODELS

def distancewithoutnormalise(bin_image):
    res = np.zeros_like(bin_image)
    for j in range(1, bin_image.max() + 1):
        one_cell = np.zeros_like(bin_image)
        one_cell[bin_image == j] = 1
        one_cell = distance_transform_cdt(one_cell)
        res[bin_image == j] = one_cell[bin_image == j]
    res = res.astype("uint8")
    return res
#  ann_dist = distancewithoutnormalise(ann_inst)

@weighted_loss
def my_loss(pred, target):
    pred = pred.squeeze()
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss


@MODELS.register_module()
class MyLoss(nn.Module):
    def __init__(
        self, reduction="mean", use_sigmoid=False, loss_weight=1.0, loss_name="MyLoss"
    ):
        """
        use_sigmoid 只是为了统一接口，没有意义
        """
        super(MyLoss, self).__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name
        self.loss_name = loss_name

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        ignore_index=-100,
    ):
        """
        ignore_index 只是为了统一接口，没有意义
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * my_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss
