from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchsummary as summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2):
        super(ConvBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(0, num_layers - 1):
            layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):        
        x = self.layers(x)
        x = self.maxpool(x)

        return x


class CustomFCN8s(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # ConvBlock
        self.conv1 = ConvBlock(in_channels=3, out_channels=64, num_layers=2)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128, num_layers=2)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256, num_layers=3)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512, num_layers=3)
        self.conv5 = ConvBlock(in_channels=512, out_channels=512, num_layers=3)

        # Fully Convolution Layers(1 x 1 conv, to adjust the dimension)
        self.fc6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=1, stride=1, padding=0)
        self.fc7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, stride=1, padding=0)
        self.score = nn.Conv2d(in_channels=4096, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        self.pool3_score = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        self.pool4_score = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

        # Transposed Convolution Layers
        self.upscore1 = nn.ConvTranspose2d(in_channels=num_classes, 
                                           out_channels=num_classes, 
                                           kernel_size=4, 
                                           stride=2, 
                                           padding=1)
        self.upscore2 = nn.ConvTranspose2d(in_channels=num_classes, 
                                           out_channels=num_classes, 
                                           kernel_size=4, 
                                           stride=2, 
                                           padding=1)
        self.upscore3 = nn.ConvTranspose2d(in_channels=num_classes, 
                                           out_channels=num_classes, 
                                           kernel_size=16,
                                           stride=8, 
                                           padding=4)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d()

        self.init_weights()

    def init_weights(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    self.normal_init(m)
            except:
                self.normal_init(block)

    def normal_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_uniform_(tensor=m.weight)
            m.bias.data.fill_(0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        out_pool3 = x
        
        x = self.conv4(x)
        out_pool4 = x

        x = self.conv5(x)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)

        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        x = self.score(x)
        x = self.upscore1(x)

        out_pool4 = self.pool4_score(out_pool4)
        x += out_pool4

        x = self.upscore2(x)        
        out_pool3 = self.pool3_score(out_pool3)
        x += out_pool3

        x = self.upscore3(x)

        return x


class CustomFCN16s(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # ConvBlock
        self.conv1 = ConvBlock(in_channels=3, out_channels=64, num_layers=2)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128, num_layers=2)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256, num_layers=3)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512, num_layers=3)
        self.conv5 = ConvBlock(in_channels=512, out_channels=512, num_layers=3)

        # Fully Convolution Layers(1 x 1 conv, to adjust the dimension)
        self.fc6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=1, stride=1, padding=0)
        self.fc7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, stride=1, padding=0)
        self.score = nn.Conv2d(in_channels=4096, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        self.pool4_score = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

        # Transposed Convolution Layers
        self.upscore1 = nn.ConvTranspose2d(in_channels=num_classes, 
                                           out_channels=num_classes,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)
        self.upscore2 = nn.ConvTranspose2d(in_channels=num_classes,
                                           out_channels=num_classes,
                                           kernel_size=32,
                                           stride=16,
                                           padding=8)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d()

        self.init_weights()

    def init_weights(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    self.normal_init(m)
            except:
                self.normal_init(block)

    def normal_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_uniform_(tensor=m.weight)
            m.bias.data.fill_(0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.conv4(x)
        out_pool4 = x

        x = self.conv5(x)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)

        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        x = self.score(x)
        x = self.upscore1(x)

        out_pool4 = self.pool4_score(out_pool4)
        x += out_pool4

        x = self.upscore2(x)

        return x


class CustomFCN32s(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # ConvBlock
        self.conv1 = ConvBlock(in_channels=3, out_channels=64, num_layers=2)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128, num_layers=2)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256, num_layers=3)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512, num_layers=3)
        self.conv5 = ConvBlock(in_channels=512, out_channels=512, num_layers=3)

        # Fully Convolution Layers(1 x 1 conv, to adjust the dimension)
        self.fc6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=1, stride=1, padding=0)
        self.fc7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, stride=1, padding=0)
        self.score = nn.Conv2d(in_channels=4096, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

        # Transposed Convolution Layers
        self.upscore = nn.ConvTranspose2d(in_channels=num_classes, 
                                          out_channels=num_classes, 
                                          kernel_size=64, 
                                          stride=32, 
                                          padding=16)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d()

        self.init_weights()

    def init_weights(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    self.normal_init(m)
            except:
                self.normal_init(block)

    def normal_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_uniform_(tensor=m.weight)
            m.bias.data.fill_(0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)        
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.relu(self.fc6(x))
        x = self.dropout(x)

        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        x = self.score(x)
        x = self.upscore(x)

        return x