import torch
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision import models
from torch import nn
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN


class CustomBackbone(nn.Module):
    def __init__(self):
        super(CustomBackbone, self).__init__()

        # Define the layers for the backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_channels = 256

    def forward(self, x):
            # Pass the input through the backbone
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.maxpool1(x)

            x = self.conv2(x)
            x = self.relu2(x)
            x = self.maxpool2(x)

            x = self.conv3(x)
            x = self.relu3(x)
            x = self.maxpool3(x)

            x = self.conv4(x)
            x = self.relu4(x)
            x = self.maxpool4(x)

            x = self.conv5(x)
            x = self.relu5(x)
            x = self.maxpool5(x)

            x = self.conv6(x)
            x = self.relu6(x)
            x = self.maxpool6(x)

            return x


def get_model(backbone, n_classes):
    if backbone == "resnet50":
        model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)
    elif backbone == "mobilenet":
        model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)

    return model

def KD_model(n_classes):
    backbone = CustomBackbone()
    anchor_generator = AnchorGenerator(
       sizes=((32, 64, 128,256),), aspect_ratios=((0.5, 1.0, 2.0,2.5),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                    output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone,n_classes,rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

    return model
