"""Mask R-CNN model construction and checkpoint loading."""

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_model(num_classes=2, pretrained=True):
    if pretrained:
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    else:
        model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, num_classes)
    return model


def load_checkpoint(model, checkpoint, device, optimizer=None, scheduler=None):
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state))
    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    return state