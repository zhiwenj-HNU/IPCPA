# ECA_ResNet

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# ECA_Module
class eca_block(nn.Module):

    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)     # 1D convolution
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)                # active

        return x * y.expand_as(x)          # Extend the dimensions to the same


class BasicResidualSEBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, 3, padding=1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion))
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = eca_block(out_channels)

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)

        # Get the width and height of the image.
        w = residual.shape[-2]
        h = residual.shape[-1]
        # print('===========：',type(w),h)

        # h = residual.size()[-1]
        # Find the smaller ones
        # if w < h:
        #     # s=(w/2).round()
        #     # l = (h / 2).round()
        #     s = (w / 2)
        #     l = (h / 2)
        # elif w > h:
        #     s = (h / 2)
        #     l = (s / 2)
        # else:
        #     #print('=======:',h ,w)
        #     s = (h /2)
        #     l=s

        j = 0
        k=[]
        for _ in range(640):
            # print('+++++++++++++++:',s,type(s),l,type(l))
            # print('------------------residual:', residual.shape, residual.max(), residual[..., :, :])
            residual0 = residual[..., j:w, j:h]
            # print('------------------residual0:', residual0.shape,residual0.max(),residual0[..., :, :])
            squeeze0 = self.squeeze(residual0)
            # print('------------------squeeze_residual0:',squeeze0.shape,squeeze0[..., :, :])
            # squeeze = torch.zeros_like(squeeze0)
            # print('------------------squeeze0:', squeeze.shape, squeeze[..., :, :])
            # squeeze += squeeze0
            # print('------------------squeeze0:', squeeze.shape,squeeze)
            k.append(squeeze0)
            # adding up
            # print(squeeze.max())
            j += 1
            if j > (w // 2) or j > (h // 2):
                break

        squeeze=torch.cat(k,dim=0)
        # squeeze = self.squeeze(residual)
        excitation=self.excitation(squeeze)

        #squeeze = squeeze.view(squeeze.size(0), -1)
        #excitation = self.excitation(squeeze)
        #excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)

class BottleneckResidualSEBlock(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.squeeze = nn.AdaptiveAvgPool2d(1)

        self.excitation = eca_block(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):

        shortcut = self.shortcut(x)

        residual = self.residual(x)
        #print('------------------开始residual:', residual.shape,residual.max(),residual)
        squeeze = self.squeeze(residual).squeeze(-1).permute(0,2,1).contiguous()
        #print('The size after pooling：',squeeze.shape)

        #print('------------------squeeze:', squeeze.shape,squeeze[...,:,:])
        # squeeze = squeeze.view(squeeze.size(0), -1)
        # #print('------------------squeeze:', squeeze.shape)
        excitation = self.excitation(squeeze).permute(0,2,1).contiguous()
        #print('The size after ECA：',excitation.shape)
        #print('------------------excitation:', excitation.shape)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)
        #print('------------------excitation:', excitation.shape)
        #print('------------------excitation:', excitation[...,0,:])
        #print('------------------shortcut:', shortcut.shape)
        #print('------------------:', (excitation.expand_as(shortcut)).shape)
        #print('------------------:::::::', (residual * excitation.expand_as(residual)).shape)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)

class ECA_ResNet(nn.Module):

    def __init__(self, block, block_num, class_num=6):  # Number of categories
        super().__init__()
        self.in_channels = 64
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.stage1 = self._make_stage(block, block_num[0], 64, 1)
        self.stage2 = self._make_stage(block, block_num[1], 128, 2)
        self.stage3 = self._make_stage(block, block_num[2], 256, 2)
        self.stage4 = self._make_stage(block, block_num[3], 512, 2)
        self.linear = nn.Linear(self.in_channels, class_num)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        #print('The first residual block is output：', x.shape)
        x = self.stage2(x)
        #print('The second residual block is output：', x.shape)
        x = self.stage3(x)
        #print('The third residual block is output：', x.shape)
        x = self.stage4(x)
        #print('The fourth residual block is output：', x.shape)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


    def _make_stage(self, block, num, out_channels, stride):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        while num - 1:
            layers.append(block(self.in_channels, out_channels, 1))
            num -= 1
        return nn.Sequential(*layers)

def ECA_ResNet_50():
    return ECA_ResNet(BottleneckResidualSEBlock, [3, 4, 6, 3])
def ECA_ResNet_101():
    return ECA_ResNet(BottleneckResidualSEBlock, [3, 4, 23, 3])
def ECA_ResNet_152():
    return ECA_ResNet(BottleneckResidualSEBlock, [3, 8, 36, 3])
