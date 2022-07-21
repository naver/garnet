"""
GaRNet
Copyright (c) 2022-present NAVER Corp.
Apache License v2.0
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        kernel_size = 7
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        Max_pool, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat((avg_pool, Max_pool), dim=1)
        out = self.conv(out)
        return self.sig(out)


class MaskSpatialGateAttention(nn.Module):
    def __init__(self):
        super(MaskSpatialGateAttention, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        kernel_size = 7
        self.conv = nn.Conv2d(3, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(3, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.sig = nn.Sigmoid()
        self.sig2 = nn.Sigmoid()
        self.sig3 = nn.Sigmoid()

    def forward(self, x, mask):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        Max_pool, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat((avg_pool, Max_pool, mask), dim=1)
        att1 = self.conv(out)
        att2 = self.conv2(out)
        return self.sig3(self.alpha * att1 + self.beta * att2), (self.sig(att1), self.sig2(att2))


class Residual_with_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, same_shape=True, Guide=True):
        super(Residual_with_Attention, self).__init__()
        self.same_shape = same_shape
        self.Guide = Guide
        stride = 1 if same_shape else 2
        padding = dilation if same_shape else 1
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding, dilation=dilation)

        if Guide:
            self.att_layer = MaskSpatialGateAttention()
        else:
            self.att_layer = SpatialAttention()

        if not same_shape:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, mask):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)

        if self.Guide:
            att, att_set = self.att_layer(out, mask)
        else:
            att = self.att_layer(out)

        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out * att + x), att_set, att


class MaskBranch(nn.Module):
    def __init__(self):
        super(MaskBranch, self).__init__()
        padding = 1
        self.initial = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.layer1_in = nn.Sequential(
            nn.MaxPool2d(2, 2),
        )
        self.layer1_out = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=padding, bias=False),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=padding, bias=False),
        )
        self.layer2_in = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=padding, bias=False),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=padding, bias=False),
        )
        self.layer2_out = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=padding, bias=False),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=padding, bias=False),
        )
        self.layer3_in = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=padding, bias=False),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=padding, bias=False),
        )
        self.layer3_out = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=padding, bias=False),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=padding, bias=False),
        )
        self.layer4_in = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=padding, bias=False),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=padding, bias=False),
        )
        self.layer4_out = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=padding, bias=False),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=padding, bias=False),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=padding, bias=False),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=padding, bias=False),
        )
        self.initial.apply(self.init_weight)
        self.initial.requires_grad = False

        self.layer1_in.apply(self.init_weight)
        self.layer1_in.requires_grad = False
        self.layer1_out.apply(self.init_weight)
        self.layer1_out.requires_grad = False

        self.layer2_in.apply(self.init_weight)
        self.layer2_in.requires_grad = False
        self.layer2_out.apply(self.init_weight)
        self.layer2_out.requires_grad = False

        self.layer3_in.apply(self.init_weight)
        self.layer3_in.requires_grad = False
        self.layer3_out.apply(self.init_weight)
        self.layer3_out.requires_grad = False

        self.layer4_in.apply(self.init_weight)
        self.layer4_in.requires_grad = False
        self.layer4_out.apply(self.init_weight)
        self.layer4_out.requires_grad = False

        self.layer5.apply(self.init_weight)
        self.layer5.requires_grad = False

    def init_weight(self, m):
        if type(m) == nn.Conv2d:
            m.weight.data.fill_(1 / (m.weight.size(3) ** 2))

    def forward(self, m):
        initial = self.initial(m)
        initial = initial / (initial.max() + 1e-6)
        in1 = self.layer1_in(initial)
        in1 = in1 / (in1.max() + 1e-6)
        out1 = self.layer1_out(in1)
        out1 = out1 / (out1.max() + 1e-6)

        in2 = self.layer2_in(out1)
        in2 = in2 / (in2.max() + 1e-6)
        out2 = self.layer2_out(in2)
        out2 = out2 / (out2.max() + 1e-6)

        in3 = self.layer3_in(out2)
        in3 = in3 / (in3.max() + 1e-6)
        out3 = self.layer3_out(in3)
        out3 = out3 / (out3.max() + 1e-6)

        in4 = self.layer4_in(out3)
        in4 = in4 / (in4.max() + 1e-6)
        out4 = self.layer4_out(in4)
        out4 = out4 / (out4.max() + 1e-6)

        out5 = self.layer5(out4)
        out5 = out5 / (out5.max() + 1e-6)
        return (in1, out1), (in2, out2), (in3, out3), (in4, out4), out5


class GaRNet(nn.Module):
    def __init__(self, in_channels):
        super(GaRNet, self).__init__()
        ndf = 64
        self.mask_branch = MaskBranch()
        for param in self.mask_branch.parameters():
            param.requires_grad = False

        self.layer1 = nn.Conv2d(4, ndf, kernel_size=4, stride=2, padding=1)
        self.layer1_lc = nn.Sequential(
            LC(ndf, ndf),
            nn.Conv2d(int(np.ceil(ndf / 2)), ndf, kernel_size=1),
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.layer1_block = Residual_with_Attention(ndf, ndf)

        self.layer2 = Residual_with_Attention(ndf, ndf)
        self.layer2_lc = nn.Sequential(
            LC(ndf, ndf),
            nn.Conv2d(int(np.ceil(ndf / 2)), ndf, kernel_size=1),
        )
        self.layer2_block = Residual_with_Attention(ndf, ndf * 2, same_shape=False)

        self.layer3 = Residual_with_Attention(ndf * 2, ndf * 2)
        self.layer3_lc = nn.Sequential(
            LC(ndf * 2, ndf * 2),
            nn.Conv2d(int(np.ceil(ndf * 2 / 2)), ndf * 2, kernel_size=1),
        )
        self.layer3_block = Residual_with_Attention(ndf * 2, ndf * 4, same_shape=False)

        self.layer4 = Residual_with_Attention(ndf * 4, ndf * 4)
        self.layer4_lc = nn.Sequential(
            LC(ndf * 4, ndf * 4),
            nn.Conv2d(int(np.ceil(ndf * 4 / 2)), ndf * 4, kernel_size=1),
        )
        self.layer4_block = Residual_with_Attention(ndf * 4, ndf * 8, same_shape=False)

        self.layer5 = Residual_with_Attention(ndf * 8, ndf * 8)

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

        self.convs_1 = nn.Conv2d(ndf, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.uplayer4 = nn.Sequential(
            nn.ConvTranspose2d(ndf, ndf, kernel_size=4, stride=2, padding=1), nn.ELU(alpha=1.0)
        )
        self.elu4 = nn.ELU(alpha=1.0)

        self.convs_2 = nn.Conv2d(ndf, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.uplayer5 = nn.Sequential(
            nn.ConvTranspose2d(ndf, in_channels, kernel_size=4, stride=2, padding=1), nn.ELU(alpha=1.0)
        )

    def forward(self, x):
        mask = x[:, 3].unsqueeze(1)
        m1, m2, m3, m4, mask = self.mask_branch(mask)
        down1 = self.layer1(x)
        down1_lc = self.layer1_lc(down1)
        down1 = self.pool(down1)
        down1, att1_set, att1 = self.layer1_block(down1, m1[1])

        down2, att2_set, att2 = self.layer2(down1, m2[0])
        down2_lc = self.layer2_lc(down2)
        down2, att3_set, att3 = self.layer2_block(down2, m2[1])

        down3, att4_set, att4 = self.layer3(down2, m3[0])
        down3_lc = self.layer3_lc(down3)
        down3, att5_set, att5 = self.layer3_block(down3, m3[1])

        down4, att6_set, att6 = self.layer4(down3, m4[0])
        down4_lc = self.layer4_lc(down4)
        down4, att7_set, att7 = self.layer4_block(down4, m4[1])

        down5, att8_set, att8 = self.layer5(down4, mask)
        down6 = self.layer6(down5)

        up1 = self.elu1(down4_lc + self.uplayer1(down6))
        up2 = self.elu2(down3_lc + self.uplayer2(up1))
        up3 = self.elu3(down2_lc + self.uplayer3(up2))
        up3_o = self.convs_1(up3)

        up4 = self.elu4(down1_lc + self.uplayer4(up3))
        up4_o = self.convs_2(up4)

        up5 = self.uplayer5(up4)
        return (
            up1,
            up2,
            up3_o,
            up4_o,
            up5,
            (m1[1], m2[0], m2[1], m3[0], m3[1], m4[0], m4[1], mask),
            (att1_set, att2_set, att3_set, att4_set, att5_set, att6_set, att7_set, att8_set),
            (att1, att2, att3, att4, att5, att6, att7, att8),
        )


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


if __name__ == "__main__":

    model = GaRNet(3)
    x = torch.ones((1, 4, 512, 512))
    _, _, _, _, result, _, _ = model(x)
    print(result.shape)
