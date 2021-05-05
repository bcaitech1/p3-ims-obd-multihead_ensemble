import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import vgg16

import segmentation_models_pytorch as smp

class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s,self).__init__()
        self.pretrained_model = vgg16(pretrained = True)
        features, classifiers = list(self.pretrained_model.features.children()), list(self.pretrained_model.classifier.children())

        self.features_map1 = nn.Sequential(*features[0:17])
        self.features_map2 = nn.Sequential(*features[17:24])
        self.features_map3 = nn.Sequential(*features[24:31])
        
        # Score pool3
        self.score_pool3_fr = nn.Conv2d(256, num_classes, 1)
        
        # Score pool4        
        self.score_pool4_fr = nn.Conv2d(512, num_classes, 1)        
        
        # fc6 ~ fc7
        self.conv = nn.Sequential(nn.Conv2d(512, 4096, kernel_size = 1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(),
                                  nn.Conv2d(4096, 4096, kernel_size = 1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout()
                                  )
        
        # Score
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size = 1)
        
        # UpScore2 using deconv
        self.upscore2 = nn.ConvTranspose2d(num_classes,
                                           num_classes,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)
        
        # UpScore2_pool4 using deconv
        self.upscore2_pool4 = nn.ConvTranspose2d(num_classes, 
                                                 num_classes, 
                                                 kernel_size=4,
                                                 stride=2,
                                                 padding=1)
        
        # UpScore8 using deconv
        self.upscore8 = nn.ConvTranspose2d(num_classes, 
                                           num_classes,
                                           kernel_size=16,
                                           stride=8,
                                           padding=4)
    
    def forward(self, x):
        pool3 = h = self.features_map1(x)
        pool4 = h = self.features_map2(h)
        h = self.features_map3(h)
        
        h = self.conv(h)
        h = self.score_fr(h)
       
        score_pool3c = self.score_pool3_fr(pool3)    
        score_pool4c = self.score_pool4_fr(pool4)
        
        # Up Score I
        upscore2 = self.upscore2(h)
        
        # Sum I
        h = upscore2 + score_pool4c
        
        # Up Score II
        upscore2_pool4c = self.upscore2_pool4(h)
        
        # Sum II
        h = upscore2_pool4c + score_pool3c
        
        # Up Score III
        upscore8 = self.upscore8(h)
        
        return upscore8


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=12):
        super(DeepLabV3, self).__init__()
        self.model = smp.DeepLabV3(
            encoder_name='efficientnet-b0',
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes,
        )
    def forward(self, x):
        return self.model(x)

class DeepLabv3Plus(nn.Module):
    def __init__(self, num_classes=12):
        super(DeepLabv3Plus, self).__init__()
        self.model = smp.DeepLabV3Plus(encoder_name="efficientnet-b3",
                encoder_weights="imagenet",
                in_channels=3,
                classes=12,
                )
    def forward(self, x):
        return self.model(x)

class UnetPlusPlus(nn.Module):
    def __init__(self, num_classes=12):
        super(UnetPlusPlus, self).__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            )
    def forward(self, x):
        return self.model(x)

class PSPNet(nn.Module):
    def __init__(self, num_classes=12):
        super(PSPNet, self).__init__()
        self.model = smp.PSPNet(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=3,
            psp_out_channels=512, # A number of filters in Spatial Pyramid
            psp_use_batchnorm=True,
            psp_dropout=0.2,
            classes=num_classes,
            activation=None, 
            upsampling=8, 
            aux_params=None)
    def forward(self, x):
        return self.model(x)