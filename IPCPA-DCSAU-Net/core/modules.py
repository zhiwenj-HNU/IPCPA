import math

import torch.nn as nn
import torch
###########################################################################################################
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
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=self.kernel_size(),
            padding=(self.kernel_size() - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels) / self.gamma) + self.b / self.gamma))
        out = k if k % 2 else k + 1
        #print('eca卷积和大小：',out)
        return out

    def forward(self, x):

        # feature descriptor on the global spatial information
        #y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y
###########################################################################################################

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding,IPCPA="False"):
        super(ResidualConv, self).__init__()
        self.IPCPA=IPCPA   ##################################################################
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.rule1 = nn.ReLU()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(output_dim)
        self.rule2 = nn.ReLU()
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)

        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )
        #########################################
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.squeeze1 = nn.AdaptiveMaxPool2d(1)  #########################################################################
        self.eca = eca_layer(input_dim * 4)
        self.a = torch.rand(1)
        self.e = torch.tensor([0.751])  ############################################################################
        ##############################################
    def forward(self, x):
        out=self.bn1(x)
        out=self.rule1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.rule2(out)
        out = self.conv2(out)
        ##########################################################################################
        #print('IPCPA:', self.IPCPA)  ####################################################
        if self.IPCPA == 'True':
            # 获取图像宽高
            # print('特征图开始缩小前的大小：',out.shape)
            b, d, h, w = out.size()[0], out.size()[1], out.size()[-2], out.size()[-1]
            i = 1

            for j in range(64):
                if w == 2 * j or h == 2 * j:
                    break
                residual0 = out[..., j:w - j, j:h - j]  # 特征图逐渐池化
                # print('特征图逐步缩小：',residual0.shape)
                if self.a > self.e:
                    squeeze0 = self.squeeze(residual0)
                else:
                    squeeze0 = self.squeeze1(residual0)
                i += 1
                # print('特征图池化操作:',squeeze0.shape)
                squeeze = torch.zeros_like(squeeze0)
                squeeze += squeeze0

            sigmoid = self.eca(squeeze)
            # print('eca模块输出大小：', out.shape)
            out = out * sigmoid.expand_as(out)
        #########################################################################################

        return out + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2
