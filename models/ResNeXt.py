# ResNeXt

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


CARDINALITY = 32
DEPTH = 4
BASEWIDTH = 64

class ResNextBottleNeckC(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        E = CARDINALITY                           # The groups of the feature maps were divided into E groups

        C = int(DEPTH * out_channels / BASEWIDTH)  #  number of channels per group
        self.split_transforms = nn.Sequential(
            nn.Conv2d(in_channels, E * C, kernel_size=1, groups=E, bias=False),
            nn.BatchNorm2d(E * C),
            nn.ReLU(inplace=True),
            nn.Conv2d(E * C, E * C, kernel_size=3, stride=stride, groups=E, padding=1, bias=False),
            nn.BatchNorm2d(E * C),
            nn.ReLU(inplace=True),
            nn.Conv2d(E * C, out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * 4))

    def forward(self, x):
        return F.relu(self.split_transforms(x) + self.shortcut(x))

class ResNext(nn.Module):
    def __init__(self, block, num_blocks, class_names=10):  # Number of categories, modified as needed
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2 = self._make_layer(block, num_blocks[0], 64, 1)
        self.conv3 = self._make_layer(block, num_blocks[1], 128, 2)
        self.conv4 = self._make_layer(block, num_blocks[2], 256, 2)
        self.conv5 = self._make_layer(block, num_blocks[3], 512, 2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, class_names)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_block, out_channels, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)

def ResNeXt_50():
    return ResNext(ResNextBottleNeckC, [3, 4, 6, 3])
def ResNeXt_101():
    return ResNext(ResNextBottleNeckC, [3, 4, 23, 3])



