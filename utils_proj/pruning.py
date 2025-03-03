import torch
import torch.nn.utils.prune as prune
from torchvision.models import detection
from torch import nn
import os
from torchvision import models
import time

def global_pruning(backbone, pruning_ratio):
    model = torch.load(os.path.join('code/models', backbone, 'model_20ep.pth'), map_location='cpu')
    if backbone == "resnet50":
        pruned_model = models.detection.fasterrcnn_resnet50_fpn_v2()
        in_features = pruned_model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
        pruned_model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, 3)
    elif backbone == "mobilenet":
        pruned_model = models.detection.fasterrcnn_mobilenet_v3_large_fpn()
        in_features = pruned_model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
        pruned_model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, 3)
    pruned_model.load_state_dict(model.state_dict())
    print('Starting pruning...')
    start = time.time()
    modules_to_prune = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            modules_to_prune.append((name, module))
    for name, module in modules_to_prune:
        prune.l1_unstructured(module, name='weight', amount=pruning_ratio)

# Remove the pruning reparameterization buffers
    for name, module in modules_to_prune:
        prune.remove(module, 'weight')
    end = time.time()
    print('Pruning time: {:.2f} seconds'.format(end-start))
    return pruned_model