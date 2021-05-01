import torch
import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp

class Effi_Unet_NS(nn.Module):  # Unet(EfficientNetB0 with noisy student)
    def __init__(self, num_classes=12):
        super(Effi_Unet_NS, self).__init__()
        self.model = smp.Unet(
            encoder_name="timm-efficientnet-b0",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="noisy-student",
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes = num_classes,  # model output channels (number of classes in your dataset)
        )
    def forward(self,x):
        return self.model(x)

class UnetPlusPlus(nn.Module):  # Unet(EfficientNetB0 with noisy student)
    def __init__(self, num_classes=12):
        super(UnetPlusPlus, self).__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="noisy-student",
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes = num_classes,  # model output channels (number of classes in your dataset)
        )
    def forward(self,x):
        return self.model(x)
        
class DeepLabV3(nn.Module):
    def __init__(self, num_classes=12):
        super(DeepLabV3, self).__init__()
        self.model = smp.DeepLabV3(
            encoder_name='efficientnet-b0',
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes
        )
    def forward(self,x):
        return self.model(x)

    
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=12):
        super(DeepLabV3Plus, self).__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name='efficientnet-b0',
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes
        )
    def forward(self,x):
        return self.model(x)
    
    
class PSPNet(nn.Module):
    def __init__(self, num_classes=12):
        super(PSPNet, self).__init__()
        self.model = smp.PSPNet(
            encoder_name='efficientnet-b0',
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes
        )
    def forward(self,x):
        return self.model(x)


if __name__ == "__main__":
    import torch
    from torchvision.models import vgg16

    backbone = vgg16(pretrained=False)
    model = FCN8s(backbone=backbone, num_cls=12)

    with torch.no_grad():
        tmp_input = torch.zeros((2, 3, 512, 512))
        tmp_output = model(tmp_input)
        print(tmp_output.shape)
