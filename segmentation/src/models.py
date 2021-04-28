import torch
import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp

class FCN8s(nn.Module):
    def __init__(self, num_classes=12):
        super(FCN8s, self).__init__()

        self.backbone = models.vgg16(pretrained=True)
        features = list(self.backbone.features.children())
        classifiers = list(self.backbone.classifier.children())

        self.features_map1 = nn.Sequential(*features[0:17])
        self.features_map2 = nn.Sequential(*features[17:24])
        self.features_map3 = nn.Sequential(*features[24:31])

        # Score pool3
        self.score_pool3_fr = nn.Conv2d(256, num_classes, 1)

        # Score pool4
        self.score_pool4_fr = nn.Conv2d(512, num_classes, 1)

        # fc6 ~ fc7
        self.conv = nn.Sequential(nn.Conv2d(512, 4096, kernel_size=1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(),
                                  nn.Conv2d(4096, 4096, kernel_size=1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout()
                                  )

        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

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

class Unet(nn.Module):
    def __init__(self, num_classes=12):
        super(Unet, self).__init__()
        self.model = smp.Unet(
            encoder_name="efficientnet-b0",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",
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
    
####################################################
class SegNet(nn.Module):
    def __init__(self, num_classes=12, init_weights=True):
        super(SegNet, self).__init__()
        backbone = models.vgg16_bn(pretrained = True)
        
        self.enc_conv1 = nn.Sequential(*list(backbone.features.children())[0:4])
        self.enc_conv2 = nn.Sequential(*list(backbone.features.children())[7:13])
        self.enc_conv3 = nn.Sequential(*list(backbone.features.children())[14:23])
        self.enc_conv4 = nn.Sequential(*list(backbone.features.children())[24:33])
        self.enc_conv5 = nn.Sequential(*list(backbone.features.children())[34:43])
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2 , return_indices = True)
        self.dec_conv5 = self.make_CB(3, 512, 512)
        self.dec_conv4 = self.make_CB(3, 512, 256)
        self.dec_conv3 = self.make_CB(3, 256, 128)
        self.dec_conv2 = self.make_CB(2, 128, 64)
        self.dec_conv1 = self.make_CB(1, 64, 64)
        self.unpool = nn.MaxUnpool2d(kernel_size = 2, stride = 2 )
        
        self.score = nn.Conv2d(64, num_classes, kernel_size =3 , stride = 1, padding = 1)
        
    def forward(self, x):
        x = self.enc_conv1(x)
        x , pool1 = self.pool(x)
        x = self.enc_conv2(x)
        x , pool2 = self.pool(x)
        x = self.enc_conv3(x)
        x , pool3 = self.pool(x)
        x = self.enc_conv4(x)
        x , pool4 = self.pool(x)
        x = self.enc_conv5(x)
        x , pool5 = self.pool(x)
        x = self.unpool(x , pool5)
        x = self.dec_conv5(x)
        x = self.unpool(x , pool4)
        x = self.dec_conv4(x)
        x = self.unpool(x , pool3)
        x = self.dec_conv3(x)
        x = self.unpool(x , pool2)
        x = self.dec_conv2(x)
        x = self.unpool(x , pool1)
        x = self.dec_conv1(x)
        x = self.score(x)
        return x 
    
    def make_CB(self, repeat, in_channels , out_channels, kernel_size = 3 , stride = 1 , padding = 1):
        layers = []
        for i in range(repeat):
            if (i == repeat-1):
                layers.append(nn.Conv2d(in_channels, out_channels,kernel_size, stride = stride , padding = padding))
                layers.append(nn.BatchNorm2d(out_channels))
            else :
                layers.append(nn.Conv2d(in_channels, in_channels,kernel_size, stride = stride , padding = padding))
                layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.ReLU(True))

        return nn.Sequential(*layers)

if __name__ == "__main__":
    import torch
    from torchvision.models import vgg16

    backbone = vgg16(pretrained=False)
    model = FCN8s(backbone=backbone, num_cls=12)

    with torch.no_grad():
        tmp_input = torch.zeros((2, 3, 512, 512))
        tmp_output = model(tmp_input)
        print(tmp_output.shape)

