# IPC_ResNet
# The same layer feature layer is alternated between avg pooling and max pooling

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo
#from .eca_module import eca_layer


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channels: Number of channels in the input tensor
        b: Hyper-parameter for adaptive kernel size formulation. Default: 1
        gamma: Hyper-parameter for adaptive kernel size formulation. Default: 2
    """
    def __init__(self, channels, b=1, gamma=2):
        super(eca_layer, self).__init__()

        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size(), padding=(self.kernel_size() - 1) // 2, bias=False,)  # 1D convolution setup
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels) / self.gamma) + self.b / self.gamma))
        out = k if k % 2 else k + 1
        #print('eca convolution kernel size：',out)
        return out

    def forward(self, x):
        y = self.conv(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)   # activation

        return y

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ECA_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ECA_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)
        if self.downsample is not None:  # downsample
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ECA_Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ECA_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.squeeze1 = nn.AdaptiveMaxPool2d(1)
        self.eca = eca_layer(planes * 4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out1 = self.bn3(out)
        # to get the image width and height
        # print('The size of the feature map before it starts to shrink:',out.shape)
        b, d, h, w = out.size()[0], out.size()[1], out.size()[-2], out.size()[-1]
        i=1
        for j in range(64):
            if w == 2 * j or h == 2 * j:
                break
            residual0 = out[..., j:w - j, j:h - j]  # feature maps  gradually pooling
            # print('feature map gradually reduce:',residual0.shape)
            if i%2==1:
                squeeze0 = self.squeeze(residual0)
            else:
                squeeze0=self.squeeze1(residual0)
            i+=1
            # print('Feature map pooling operation:',squeeze0.shape)
            squeeze = torch.zeros_like(squeeze0)
            squeeze += squeeze0

        sigmoid = self.eca(squeeze)
        #print('ECA module output size:', out.shape)
        out=out1*sigmoid.expand_as(out1)
        if self.downsample is not None:
            residual = self.downsample(x)
        #print('The size of the two ways before merging out, residual:', out.shape, residual.shape)
        out += residual
        out = self.relu(out)

        return out

class IPC_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):  # Number of categories, modified as needed.
        self.inplanes = 64
        super(IPC_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False,),nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        x = self.layer1(x)
        #print('The first residual block outputs:', x.shape)
        x = self.layer2(x)
        #print('The second residual block outputs:', x.shape)
        x = self.layer3(x)
        #print('The third residual block outputs:', x.shape)
        x = self.layer4(x)
        #print('The fourth residual block outputs:',x.shape)
        #x = self.avgpool(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def IPC_ResNet_18():
    return IPC_ResNet(ECA_Bottleneck, [2, 2, 2, 2])
def IPC_ResNet_34():
    return IPC_ResNet(ECA_Bottleneck, [3, 4, 6, 3])
def IPC_ResNet_50():
    return IPC_ResNet(ECA_Bottleneck, [3, 4, 6, 3])
def IPC_ResNet_101():
    return IPC_ResNet(ECA_Bottleneck, [3, 4, 23, 3])






