import math

import torch
import torch.nn as nn
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

class PreActivateDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,IPCPA="False"):
        self.IPCPA=IPCPA   ########################################################
        super(PreActivateDoubleConv, self).__init__()
        self.bn1=nn.BatchNorm2d(in_channels)
        self.relu1=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 =nn.BatchNorm2d(out_channels)
        self.relu2=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        #########################################
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.squeeze1 = nn.AdaptiveMaxPool2d(1)  #########################################################################
        self.eca = eca_layer(in_channels * 4)
        self.a = torch.rand(1)
        self.e = torch.tensor([0.7485])  ############################################################################
        ##############################################

    def forward(self, x):
        x=self.bn1(x)
        out=self.relu1(x)
        out=self.conv1(out)
        out1 = self.bn2(out)
        ##########################################################################################
        if self.IPCPA == 'True':
            # print('IPCPA:',self.IPCPA)  ####################################################
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
            out = out1 * sigmoid.expand_as(out1)
        #########################################################################################

        out = self.relu2(out)
        out = self.conv2(out)
# ##########################################################################################
#         if self.IPCPA=='True':
#             #print('IPCPA:',self.IPCPA)  ####################################################
#             # 获取图像宽高
#             # print('特征图开始缩小前的大小：',out.shape)
#             b, d, h, w = out.size()[0], out.size()[1], out.size()[-2], out.size()[-1]
#             i = 1
#
#             for j in range(64):
#                 if w == 2 * j or h == 2 * j:
#                     break
#                 residual0 = out[..., j:w - j, j:h - j]  # 特征图逐渐池化
#                 # print('特征图逐步缩小：',residual0.shape)
#                 if self.a > self.e:
#                     squeeze0 = self.squeeze(residual0)
#                 else:
#                     squeeze0 = self.squeeze1(residual0)
#                 i += 1
#                 # print('特征图池化操作:',squeeze0.shape)
#                 squeeze = torch.zeros_like(squeeze0)
#                 squeeze += squeeze0
#
#             sigmoid = self.eca(squeeze)
#             # print('eca模块输出大小：', out.shape)
#             out = out * sigmoid.expand_as(out)
# #########################################################################################
        return out

class PreActivateResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels,IPCPA="False"):
        super(PreActivateResUpBlock, self).__init__()
        self.IPCPA=IPCPA   ####################################################################
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)    # upsample 是上采样，直观理解就是放大图像，采用各种插值算法来扩充 feature map
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.double_conv = PreActivateDoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x) + self.ch_avg(x)

class PreActivateResBlock(nn.Module):   ######################################################1
    def __init__(self, in_channels, out_channels,IPCPA="False"):
        self.IPCPA=IPCPA   ######################################################
        super(PreActivateResBlock, self).__init__()
        self.cov1=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)

        self.double_conv = PreActivateDoubleConv(in_channels, out_channels, self.IPCPA)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        out0 = self.cov1(x)
        identity=self.bn1(out0)

        out = self.double_conv(x)
        out = out + identity
        return self.down_sample(out), out

class DoubleConv(nn.Module):                                                    ###### resunet
    def __init__(self, in_channels, out_channels,IPCPA="False"):
        super(DoubleConv, self).__init__()
        self.IPCPA=IPCPA  ##################################################
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 =nn.BatchNorm2d(out_channels)
        self.relu2=nn.ReLU(inplace=True)
        #########################################
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.squeeze1 = nn.AdaptiveMaxPool2d(1)  #########################################################################
        self.eca = eca_layer(in_channels * 4)
        self.a = torch.rand(1)
        self.e = torch.tensor([0.7478])  ############################################################################
        ##############################################

    def forward(self, x):
        out = self.conv1(x)
        # if self.IPCPA == 'True':
        #     # 获取图像宽高
        #     # print('特征图开始缩小前的大小：',out.shape)
        #     b, d, h, w = x.size()[0], x.size()[1], x.size()[-2], x.size()[-1]
        #     i = 1
        #
        #     for j in range(64):
        #         if w == 2 * j or h == 2 * j:
        #             break
        #         residual0 = x[..., j:w - j, j:h - j]  # 特征图逐渐池化
        #         # print('特征图逐步缩小：',residual0.shape)
        #         if self.a > self.e:
        #             squeeze0 = self.squeeze(residual0)
        #         else:
        #             squeeze0 = self.squeeze1(residual0)
        #         i += 1
        #         # print('特征图池化操作:',squeeze0.shape)
        #         squeeze = torch.zeros_like(squeeze0)
        #         squeeze += squeeze0
        #
        #     sigmoid = self.eca(squeeze)
        #     # print('eca模块输出大小：', out.shape)
        #     out = out * sigmoid.expand_as(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out1=self.bn2(out)
        # ##########################################################################################
        # print('IPCPA:', self.IPCPA)  ####################################################
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
            out = out1 * sigmoid.expand_as(out1)
        # #########################################################################################
        #out = self.bn2(out)
        out = self.relu2(out)

        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,IPCPA="False"):
        super(ResBlock, self).__init__()
        self.IPCPA=IPCPA  ##################################
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.double_conv = DoubleConv(in_channels, out_channels,IPCPA=IPCPA)
        self.down_sample = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.double_conv(x)
        out = self.relu(out + identity)
        return self.down_sample(out), out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels,IPCPA='False'):
        super(UpBlock, self).__init__()
        self.IPCPA=IPCPA  ########################
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(in_channels, out_channels,IPCPA=IPCPA)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, out_classes=1, IPCPA='False'):
        super(UNet, self).__init__()
        self.IPCPA = IPCPA  #######################################################
        self.down_conv1 = DownBlock(3, 64)   ################################################################
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)

        self.double_conv = DoubleConv(512, 1024)

        self.up_conv4 = UpBlock(512 + 1024, 512)
        self.up_conv3 = UpBlock(256 + 512, 256)
        self.up_conv2 = UpBlock(128 + 256, 128)
        self.up_conv1 = UpBlock(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x

class DeepResUNet(nn.Module):
    def __init__(self, out_classes=1, IPCPA='False'):
        super(DeepResUNet, self).__init__()
        self.IPCPA=IPCPA  #######################################################
        self.down_conv1 = PreActivateResBlock(3, 64,self.IPCPA)  ################################################################
        self.down_conv2 = PreActivateResBlock(64, 128,self.IPCPA)
        self.down_conv3 = PreActivateResBlock(128, 256,self.IPCPA)
        self.down_conv4 = PreActivateResBlock(256, 512,self.IPCPA)

        self.double_conv = PreActivateDoubleConv(512, 1024,self.IPCPA)

        self.up_conv4 = PreActivateResUpBlock(512 + 1024, 512,self.IPCPA)
        self.up_conv3 = PreActivateResUpBlock(256 + 512, 256,self.IPCPA)
        self.up_conv2 = PreActivateResUpBlock(128 + 256, 128,self.IPCPA)
        self.up_conv1 = PreActivateResUpBlock(128 + 64, 64,self.IPCPA)

        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        #print('x.shape:',x.shape)  #####################################################################
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x

class ResUNet(nn.Module):
    """
    Hybrid solution of resnet blocks and double conv blocks
    """
    def __init__(self, out_classes=1, IPCPA='False'):
        super(ResUNet, self).__init__()
        self.IPCPA = IPCPA  #######################################################
        self.down_conv1 = ResBlock(3, 64,self.IPCPA)  ##################################################################################
        self.down_conv2 = ResBlock(64, 128,self.IPCPA)
        self.down_conv3 = ResBlock(128, 256,self.IPCPA)
        self.down_conv4 = ResBlock(256, 512,self.IPCPA)

        self.double_conv = DoubleConv(512, 1024,self.IPCPA)

        self.up_conv4 = UpBlock(512 + 1024, 512,self.IPCPA)
        self.up_conv3 = UpBlock(256 + 512, 256,self.IPCPA)
        self.up_conv2 = UpBlock(128 + 256, 128,self.IPCPA)
        self.up_conv1 = UpBlock(128 + 64, 64,self.IPCPA)

        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x

class ONet(nn.Module):
    def __init__(self, alpha=470, beta=40, out_classes=1, IPCPA='False'):
        super(ONet, self).__init__()
        self.IPCPA = IPCPA  #######################################################
        self.alpha = alpha
        self.beta = beta
        self.down_conv1 = ResBlock(3, 64)  ############################################################################
        self.down_conv2 = ResBlock(64, 128)
        self.down_conv3 = ResBlock(128, 256)
        self.down_conv4 = ResBlock(256, 512)

        self.double_conv = DoubleConv(512, 1024)

        self.up_conv4 = UpBlock(512 + 1024, 512)
        self.up_conv3 = UpBlock(256 + 512, 256)
        self.up_conv2 = UpBlock(128 + 256, 128)
        self.up_conv1 = UpBlock(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)
        self.input_output_conv = nn.Conv2d(2, 1, kernel_size=1)


    def forward(self, inputs):
        input_tensor, bounding = inputs
        x, skip1_out = self.down_conv1(input_tensor + (bounding * self.alpha))
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        input_output = torch.cat([x, bounding * self.beta], dim=1)
        x = self.input_output_conv(input_output)
        return x