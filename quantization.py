import torchvision
import os
import torch
import copy
from utils_proj import *

#sys.path.append("/Users/sebastianodarconso/Desktop/PlayerVsReferee_final/code/detection")
class QuantizedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model_fp32 = model
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x


def print_model_size(model):
    torch.save(model.state_dict(), 'tmp.pt')
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')


def quantize_model(model):
    backend = 'x86'
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_static_quantized = torch.quantization.prepare(model, inplace=True)
    model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=True)
    return model_static_quantized

if __name__ == '__main__':

    print('model pre-quantization')
    model = torch.load('code/models/resnet50/model_20ep.pth', map_location='cpu')
    print(torch.backends.quantized.supported_engines)
    backend = 'x86'
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_static_quantized = torch.quantization.prepare(model, inplace=False)
    model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
    print_model_size(model_static_quantized) ## will print quantized model size
    model_static_quantized.eval()
    dummy = torch.randn(3, 256, 256)
    o = model_static_quantized([dummy])
    exit()
    model = QuantizedModel(model)
    model_quantized = copy.deepcopy(model)
    print_model_size(model)

    ## backbone quantization
    model_backbone =model.model_fp32.backbone
    print('quantizing backbone')
    print_model_size(model_backbone)
    backbone_quantized = quantize_model(model_backbone)
    print_model_size(backbone_quantized)

    ## roi heads quantization
    model_roi_heads = model.model_fp32.roi_heads
    print('quantizing roi head module')
    print_model_size(model_roi_heads)
    roi_quantized = quantize_model(model_roi_heads)
    print_model_size(roi_quantized)

    ## region proposal network quantization
    model_rpn = model.model_fp32.rpn
    print('quantizing RPN module')
    print_model_size(model_rpn)
    rpn_quantized = quantize_model(model_rpn)
    print_model_size(rpn_quantized)

    ## model ensemble
    print('adding all modules back')
    model_quantized.model_fp32.backbone = backbone_quantized
    model_quantized.model_fp32.roi_heads = roi_quantized
    model_quantized.model_fp32.rpn = rpn_quantized

    print('model after quantization')
    #torch.save(model_quantized, 'faster_rcnn_quantized_resnet50.pth')
    #torch.save(model_quantized.state_dict, 'quantized_resnet50_state_dict.pth')
    #print_model_size(model_quantized)

    # print(torch.backends.quantized.supported_engines)
    # print(list(model.parameters()))
    # print(list(model_quantized.parameters()))
    print(model_quantized)
    model_quantized.eval()
    dummy = torch.randn(1, 3, 256, 256)
    print(model_quantized(dummy))

