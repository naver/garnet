"""
GaRNet
Copyright (c) 2022-present NAVER Corp.
Apache License v2.0
"""

"""
We re-implemnted EnsNet by referring to HCIILAB/Scene-Text-Removal
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Discriminator(nn.Module):
    def __init__(self, in_channels=6, ndf=64, num_blocks=3):
        super(Discriminator, self).__init__()
        kernel_size = 4
        padding = 1
        self.initialblock = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.blocks = []
        mul = 1
        for i in range(1, num_blocks):
            mul_prev = mul
            mul = min(2 ** i, 8)
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(ndf * mul_prev, ndf * mul, kernel_size=kernel_size, stride=2, padding=padding),
                    nn.BatchNorm2d(ndf * mul),
                    nn.LeakyReLU(negative_slope=0.2),
                )
            )
        self.blocks = nn.Sequential(*self.blocks)
        mul_prev = mul
        mul = min(2 ** i, 8)
        self.outblock = nn.Sequential(
            nn.Conv2d(ndf * mul_prev, ndf * mul, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(ndf * mul),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(ndf * mul, 1, kernel_size=kernel_size, stride=1, padding=padding),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):

        out = self.initialblock(x)
        out = self.blocks(out)
        out = self.outblock(out)
        return self.sig(out)


class label_Discriminator(nn.Module):
    def __init__(self, in_channels, ndf=1, n_layers=3, use_sigmoid=False, use_bias=False):
        super(label_Discriminator, self).__init__()

        kernel_size = 70
        padding = 23

        self.model = nn.Conv2d(in_channels, ndf, kernel_size=kernel_size, stride=8, padding=padding, bias=use_bias)

    def forward(self, x):
        return self.model(x)


class LC(nn.Module):
    def __init__(self, in_channels, outer_channels):
        super(LC, self).__init__()
        channels = int(np.ceil(outer_channels / 2))
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, same_shape=True):
        super(Residual, self).__init__()
        self.same_shape = same_shape

        stride = 1 if same_shape else 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if not same_shape:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)


def get_vggnet(style_layers):
    vgg19 = models.vgg19(pretrained=True)
    net = []

    for i in range(max(style_layers) + 1):
        net.append(vgg19.features[i])

    return nn.Sequential(*net)


class STE(nn.Module):
    def __init__(self, in_channels, ndf=64):
        super(STE, self).__init__()

        # LC is Lateral connection
        self.layer1 = nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)
        self.layer1_lc = nn.Sequential(LC(ndf, ndf), nn.Conv2d(int(np.ceil(ndf / 2)), ndf, kernel_size=1))
        self.layer1_block = nn.Sequential(nn.MaxPool2d(2, 2), Residual(ndf, ndf))

        self.layer2 = Residual(ndf, ndf)
        self.layer2_lc = nn.Sequential(LC(ndf, ndf), nn.Conv2d(int(np.ceil(ndf / 2)), ndf, kernel_size=1))
        self.layer2_block = Residual(ndf, ndf * 2, same_shape=False)

        self.layer3 = Residual(ndf * 2, ndf * 2)
        self.layer3_lc = nn.Sequential(
            LC(ndf * 2, ndf * 2), nn.Conv2d(int(np.ceil(ndf * 2 / 2)), ndf * 2, kernel_size=1)
        )
        self.layer3_block = Residual(ndf * 2, ndf * 4, same_shape=False)

        self.layer4 = Residual(ndf * 4, ndf * 4)
        self.layer4_lc = nn.Sequential(
            LC(ndf * 4, ndf * 4), nn.Conv2d(int(np.ceil(ndf * 4 / 2)), ndf * 4, kernel_size=1)
        )
        self.layer4_block = Residual(ndf * 4, ndf * 8, same_shape=False)

        self.layer5 = Residual(ndf * 8, ndf * 8)
        self.layer6 = nn.Conv2d(ndf * 8, 2, kernel_size=1)

        self.uplayer1 = nn.Sequential(
            nn.ConvTranspose2d(2, ndf * 4, kernel_size=4, stride=2, padding=1), nn.ELU(alpha=1.0)
        )
        self.elu1 = nn.ELU(alpha=1.0)

        self.uplayer2 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 4, ndf * 2, kernel_size=4, stride=2, padding=1), nn.ELU(alpha=1.0)
        )
        self.elu2 = nn.ELU(alpha=1.0)

        self.uplayer3 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 2, ndf, kernel_size=4, stride=2, padding=1), nn.ELU(alpha=1.0)
        )
        self.elu3 = nn.ELU(alpha=1.0)

        self.convs_1 = nn.Conv2d(ndf, 3, kernel_size=1, stride=1, padding=0, bias=False)

        self.uplayer4 = nn.Sequential(
            nn.ConvTranspose2d(ndf, ndf, kernel_size=4, stride=2, padding=1), nn.ELU(alpha=1.0)
        )
        self.elu4 = nn.ELU(alpha=1.0)

        self.convs_2 = nn.Conv2d(ndf, 3, kernel_size=1, stride=1, padding=0, bias=False)

        self.uplayer5 = nn.Sequential(nn.ConvTranspose2d(ndf, 3, kernel_size=4, stride=2, padding=1), nn.ELU(alpha=1.0))

    def forward(self, x):
        down1 = self.layer1(x)
        down1_lc = self.layer1_lc(down1)
        down1 = self.layer1_block(down1)

        down2 = self.layer2(down1)
        down2_lc = self.layer2_lc(down2)
        down2 = self.layer2_block(down2)

        down3 = self.layer3(down2)
        down3_lc = self.layer3_lc(down3)
        down3 = self.layer3_block(down3)

        down4 = self.layer4(down3)
        down4_lc = self.layer4_lc(down4)
        down4 = self.layer4_block(down4)

        down5 = self.layer5(down4)
        down6 = self.layer6(down5)

        up1 = self.elu1(down4_lc + self.uplayer1(down6))
        up2 = self.elu2(down3_lc + self.uplayer2(up1))
        up3 = self.elu3(down2_lc + self.uplayer3(up2))
        up3_o = self.convs_1(up3)

        up4 = self.elu4(down1_lc + self.uplayer4(up3))
        up4_o = self.convs_2(up4)

        up5 = self.uplayer5(up4)
        return up1, up2, up3_o, up4_o, up5
