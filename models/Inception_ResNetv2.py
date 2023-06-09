# -*- coding: UTF-8 -*-
# inception_resnet_v2
import torch
import torch.nn as nn

class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class Inception_Stem(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Sequential(BasicConv2d(input_channels, 32, kernel_size=3), BasicConv2d(32, 32, kernel_size=3, padding=1), BasicConv2d(32, 64, kernel_size=3, padding=1))
        self.branch3x3_conv = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.branch7x7a = nn.Sequential(BasicConv2d(160, 64, kernel_size=1), BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)), BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(64, 96, kernel_size=3, padding=1))
        self.branch7x7b = nn.Sequential(BasicConv2d(160, 64, kernel_size=1), BasicConv2d(64, 96, kernel_size=3, padding=1))
        self.branchpoola = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branchpoolb = BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = [self.branch3x3_conv(x), self.branch3x3_pool(x)]
        x = torch.cat(x, 1)
        x = [self.branch7x7a(x), self.branch7x7b(x)]
        x = torch.cat(x, 1)
        x = [self.branchpoola(x), self.branchpoolb(x)]
        x = torch.cat(x, 1)

        return x

class InceptionA(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, 64, kernel_size=1), BasicConv2d(64, 96, kernel_size=3, padding=1), BasicConv2d(96, 96, kernel_size=3, padding=1))
        self.branch3x3 = nn.Sequential(BasicConv2d(input_channels, 64, kernel_size=1), BasicConv2d(64, 96, kernel_size=3, padding=1))
        self.branch1x1 = BasicConv2d(input_channels, 96, kernel_size=1)
        self.branchpool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1), BasicConv2d(input_channels, 96, kernel_size=1))

    def forward(self, x):
        x = [self.branch3x3stack(x), self.branch3x3(x), self.branch1x1(x), self.branchpool(x)]
        return torch.cat(x, 1)

class ReductionA(nn.Module):
    def __init__(self, input_channels, k, l, m, n):
        super().__init__()
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, k, kernel_size=1), BasicConv2d(k, l, kernel_size=3, padding=1), BasicConv2d(l, m, kernel_size=3, stride=2))
        self.branch3x3 = BasicConv2d(input_channels, n, kernel_size=3, stride=2)
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.output_channels = input_channels + n + m

    def forward(self, x):
        x = [self.branch3x3stack(x), self.branch3x3(x), self.branchpool(x)]
        return torch.cat(x, 1)

class InceptionB(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branch7x7stack = nn.Sequential(BasicConv2d(input_channels, 192, kernel_size=1), BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(192, 224, kernel_size=(7, 1), padding=(3, 0)), BasicConv2d(224, 224, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0)))
        self.branch7x7 = nn.Sequential(BasicConv2d(input_channels, 192, kernel_size=1), BasicConv2d(192, 224, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0)))
        self.branch1x1 = BasicConv2d(input_channels, 384, kernel_size=1)
        self.branchpool = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1), BasicConv2d(input_channels, 128, kernel_size=1))

    def forward(self, x):
        x = [self.branch1x1(x), self.branch7x7(x), self.branch7x7stack(x), self.branchpool(x)]

        return torch.cat(x, 1)

class ReductionB(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branch7x7 = nn.Sequential(BasicConv2d(input_channels, 256, kernel_size=1), BasicConv2d(256, 256, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(256, 320, kernel_size=(7, 1), padding=(3, 0)), BasicConv2d(320, 320, kernel_size=3, stride=2, padding=1))
        self.branch3x3 = nn.Sequential(BasicConv2d(input_channels, 192, kernel_size=1), BasicConv2d(192, 192, kernel_size=3, stride=2, padding=1))
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = [self.branch3x3(x), self.branch7x7(x), self.branchpool(x)]

        return torch.cat(x, 1)

class InceptionC(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, 384, kernel_size=1), BasicConv2d(384, 448, kernel_size=(1, 3), padding=(0, 1)), BasicConv2d(448, 512, kernel_size=(3, 1), padding=(1, 0)),)
        self.branch3x3stacka = BasicConv2d(512, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stackb = BasicConv2d(512, 256, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3 = BasicConv2d(input_channels, 384, kernel_size=1)
        self.branch3x3a = BasicConv2d(384, 256, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3b = BasicConv2d(384, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch1x1 = BasicConv2d(input_channels, 256, kernel_size=1)
        self.branchpool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1), BasicConv2d(input_channels, 256, kernel_size=1))

    def forward(self, x):
        branch3x3stack_output = self.branch3x3stack(x)
        branch3x3stack_output = [self.branch3x3stacka(branch3x3stack_output), self.branch3x3stackb(branch3x3stack_output)]
        branch3x3stack_output = torch.cat(branch3x3stack_output, 1)
        branch3x3_output = self.branch3x3(x)
        branch3x3_output = [self.branch3x3a(branch3x3_output), self.branch3x3b(branch3x3_output)]
        branch3x3_output = torch.cat(branch3x3_output, 1)
        branch1x1_output = self.branch1x1(x)
        branchpool = self.branchpool(x)
        output = [branch1x1_output, branch3x3_output, branch3x3stack_output, branchpool]

        return torch.cat(output, 1)

class InceptionV4(nn.Module):
    def __init__(self, A, B, C, k=192, l=224, m=256, n=384, class_nums=10):
        super().__init__()
        self.stem = Inception_Stem(3)
        self.inception_a = self._generate_inception_module(384, 384, A, InceptionA)
        self.reduction_a = ReductionA(384, k, l, m, n)
        output_channels = self.reduction_a.output_channels
        self.inception_b = self._generate_inception_module(output_channels, 1024, B, InceptionB)
        self.reduction_b = ReductionB(1024)
        self.inception_c = self._generate_inception_module(1536, 1536, C, InceptionC)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout2d(1 - 0.8)
        self.linear = nn.Linear(1536, class_nums)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(-1, 1536)
        x = self.linear(x)

        return x

    @staticmethod
    def _generate_inception_module(input_channels, output_channels, block_num, block):
        layers = nn.Sequential()
        for l in range(block_num):
            layers.add_module("{}_{}".format(block.__name__, l), block(input_channels))
            input_channels = output_channels

        return layers

class InceptionResNetA(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, 32, kernel_size=1), BasicConv2d(32, 48, kernel_size=3, padding=1), BasicConv2d(48, 64, kernel_size=3, padding=1))
        self.branch3x3 = nn.Sequential(BasicConv2d(input_channels, 32, kernel_size=1), BasicConv2d(32, 32, kernel_size=3, padding=1))
        self.branch1x1 = BasicConv2d(input_channels, 32, kernel_size=1)
        self.reduction1x1 = nn.Conv2d(128, 384, kernel_size=1)
        self.shortcut = nn.Conv2d(input_channels, 384, kernel_size=1)
        self.bn = nn.BatchNorm2d(384)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = [self.branch1x1(x), self.branch3x3(x), self.branch3x3stack(x)]
        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual)
        shortcut = self.shortcut(x)
        output = self.bn(shortcut + residual)
        output = self.relu(output)

        return output

class InceptionResNetB(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branch7x7 = nn.Sequential(BasicConv2d(input_channels, 128, kernel_size=1), BasicConv2d(128, 160, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1), padding=(3, 0)))
        self.branch1x1 = BasicConv2d(input_channels, 192, kernel_size=1)
        self.reduction1x1 = nn.Conv2d(384, 1154, kernel_size=1)
        self.shortcut = nn.Conv2d(input_channels, 1154, kernel_size=1)
        self.bn = nn.BatchNorm2d(1154)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = [self.branch1x1(x),self.branch7x7(x)]
        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual) * 0.1
        shortcut = self.shortcut(x)
        output = self.bn(residual + shortcut)
        output = self.relu(output)

        return output

class InceptionResNetC(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3 = nn.Sequential(BasicConv2d(input_channels, 192, kernel_size=1), BasicConv2d(192, 224, kernel_size=(1, 3), padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1), padding=(1, 0)))
        self.branch1x1 = BasicConv2d(input_channels, 192, kernel_size=1)
        self.reduction1x1 = nn.Conv2d(448, 2048, kernel_size=1)
        self.shorcut = nn.Conv2d(input_channels, 2048, kernel_size=1)
        self.bn = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = [self.branch1x1(x),self.branch3x3(x)]
        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual) * 0.1
        shorcut = self.shorcut(x)
        output = self.bn(shorcut + residual)
        output = self.relu(output)

        return output

class InceptionResNetReductionA(nn.Module):
    def __init__(self, input_channels, k, l, m, n):
        super().__init__()
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, k, kernel_size=1), BasicConv2d(k, l, kernel_size=3, padding=1), BasicConv2d(l, m, kernel_size=3, stride=2))
        self.branch3x3 = BasicConv2d(input_channels, n, kernel_size=3, stride=2)
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.output_channels = input_channels + n + m

    def forward(self, x):
        x = [self.branch3x3stack(x),self.branch3x3(x),self.branchpool(x)]
        return torch.cat(x, 1)

class InceptionResNetReductionB(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branchpool = nn.MaxPool2d(3, stride=2)
        self.branch3x3a = nn.Sequential(BasicConv2d(input_channels, 256, kernel_size=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch3x3b = nn.Sequential(BasicConv2d(input_channels, 256, kernel_size=1), BasicConv2d(256, 288, kernel_size=3, stride=2))
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, 256, kernel_size=1), BasicConv2d(256, 288, kernel_size=3, padding=1), BasicConv2d(288, 320, kernel_size=3, stride=2))

    def forward(self, x):
        x = [self.branch3x3a(x), self.branch3x3b(x), self.branch3x3stack(x), self.branchpool(x)]
        x = torch.cat(x, 1)
        return x

class Inception_ResNetV2(nn.Module):

    def __init__(self, A, B, C, k=256, l=256, m=384, n=384, class_nums=10):
        super().__init__()
        self.stem = Inception_Stem(3)
        self.inception_resnet_a = self._generate_inception_module(384, 384, A, InceptionResNetA)
        self.reduction_a = InceptionResNetReductionA(384, k, l, m, n)
        output_channels = self.reduction_a.output_channels
        self.inception_resnet_b = self._generate_inception_module(output_channels, 1154, B, InceptionResNetB)
        self.reduction_b = InceptionResNetReductionB(1154)
        self.inception_resnet_c = self._generate_inception_module(2146, 2048, C, InceptionResNetC)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(1 - 0.8)
        self.linear = nn.Linear(2048, class_nums)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_resnet_a(x)
        x = self.reduction_a(x)
        x = self.inception_resnet_b(x)
        x = self.reduction_b(x)
        x = self.inception_resnet_c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(-1, 2048)
        x = self.linear(x)

        return x

    @staticmethod
    def _generate_inception_module(input_channels, output_channels, block_num, block):
        layers = nn.Sequential()
        for l in range(block_num):
            layers.add_module("{}_{}".format(block.__name__, l), block(input_channels))
            input_channels = output_channels
        return layers

def Inception_ResNetv2():
    return Inception_ResNetV2(5, 10, 5)