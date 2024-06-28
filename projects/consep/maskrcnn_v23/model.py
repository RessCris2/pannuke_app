import torch
import torchvision
from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_V2_Weights,
                                          MaskRCNN_ResNet50_FPN_Weights,
                                          maskrcnn_resnet50_fpn)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.resnet import ResNet50_Weights


def get_model_instance_segmentation(num_classes):
    # anchor_sizes = ((32/4,), (64/4,), (128/4,), (256/4,), (512/4,))
    # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    # anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(anchor_sizes, aspect_ratios)
    # load an instance segmentation model pre-trained on COCO
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT", weights_backbone=ResNet50_Weights.IMAGENET1K_V2)
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights_backbone=ResNet50_Weights.IMAGENET1K_V2, box_batch_size_per_image=256, box_detections_per_img=300,rpn_anchor_generator =anchor_generator)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

