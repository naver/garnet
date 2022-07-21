"""
GaRNet
Copyright (c) 2022-present NAVER Corp.
Apache License v2.0
"""

"""
We re-implemnted MTRNet by referring to MTRNet paper.
"""

import torch
import torch.nn as nn
import numpy as np


class ConvWithActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(ConvWithActivation, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        out = self.block(x)
        return out


class MTRNet(nn.Module):
    def __init__(self):
        super(MTRNet, self).__init__()
        ndf = 64
        self.conv0 = ConvWithActivation(4, ndf)
        self.conv1 = ConvWithActivation(ndf, ndf * 2)
        self.conv2 = ConvWithActivation(ndf * 2, ndf * 4)
        self.conv3 = ConvWithActivation(ndf * 4, ndf * 8)
        self.conv4 = ConvWithActivation(ndf * 8, ndf * 8)
        self.conv5 = ConvWithActivation(ndf * 8, ndf * 8)
        self.conv6 = ConvWithActivation(ndf * 8, ndf * 8)

        self.conv_t0 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(ndf * 8), nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(ndf * 8 * 2, ndf * 8, kernel_size=4, stride=1),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(),
        )
        self.conv_t1 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 8, ndf * 8, kernel_size=2, stride=2, padding=0), nn.BatchNorm2d(ndf * 8), nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(ndf * 8 * 2, ndf * 8, kernel_size=4, stride=1),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(),
        )
        self.conv_t2 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 8, ndf * 8, kernel_size=2, stride=2, padding=0), nn.BatchNorm2d(ndf * 8), nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(ndf * 8 * 2, ndf * 8, kernel_size=4, stride=1),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(),
        )
        self.conv_t3 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 8, ndf * 4, kernel_size=2, stride=2, padding=0), nn.BatchNorm2d(ndf * 4), nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(ndf * 4 * 2, ndf * 4, kernel_size=4, stride=1),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(),
        )
        self.conv_t4 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 4, ndf * 2, kernel_size=2, stride=2, padding=0), nn.BatchNorm2d(ndf * 2), nn.ReLU()
        )
        self.conv11 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(ndf * 2 * 2, ndf * 2, kernel_size=4, stride=1),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(),
        )
        self.conv_t5 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 2, ndf, kernel_size=2, stride=2, padding=0), nn.BatchNorm2d(ndf), nn.ReLU()
        )
        self.conv12 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(ndf * 2, ndf, kernel_size=4, stride=1), nn.BatchNorm2d(ndf), nn.ReLU()
        )
        self.conv_t6 = nn.Sequential(
            nn.ConvTranspose2d(ndf, ndf, kernel_size=2, stride=2, padding=0), nn.BatchNorm2d(ndf), nn.ReLU()
        )
        self.conv13 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(ndf, ndf, kernel_size=4, stride=1), nn.BatchNorm2d(ndf), nn.ReLU()
        )

        self.conv14 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(ndf, 3, kernel_size=4, stride=1), nn.Tanh())

    def forward(self, x):
        d1 = self.conv0(x)
        d2 = self.conv1(d1)
        d3 = self.conv2(d2)
        d4 = self.conv3(d3)
        d5 = self.conv4(d4)
        d6 = self.conv5(d5)
        d7 = self.conv6(d6)

        u1 = self.conv_t0(d7)
        u1 = self.conv7(torch.cat([u1, d6], dim=1))
        u2 = self.conv_t1(u1)
        u2 = self.conv8(torch.cat([u2, d5], dim=1))
        u3 = self.conv_t2(u2)
        u3 = self.conv9(torch.cat([u3, d4], dim=1))
        u4 = self.conv_t3(u3)
        u4 = self.conv10(torch.cat([u4, d3], dim=1))
        u5 = self.conv_t4(u4)
        u5 = self.conv11(torch.cat([u5, d2], dim=1))
        u6 = self.conv_t5(u5)
        u6 = self.conv12(torch.cat([u6, d1], dim=1))
        u7 = self.conv_t6(u6)
        u7 = self.conv13(u7)
        u8 = self.conv14(u7)

        return u8


class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(Discriminator, self).__init__()
        ndf = 64
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        self.conv1 = ConvWithActivation(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = ConvWithActivation(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv3 = ConvWithActivation(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)

        self.conv_t0 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(),
        )
        self.conv_t1 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 8, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(),
        )
        self.conv_t2 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 4, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(),
        )
        self.conv_t3 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 2, ndf, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(ndf), nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(ndf, 1, kernel_size=4, stride=1))

    def forward(self, x):
        d1 = self.conv0(x)
        d2 = self.conv1(d1)
        d3 = self.conv2(d2)
        d4 = self.conv3(d3)
        u1 = self.conv_t0(d4)
        u2 = self.conv_t1(u1)
        u3 = self.conv_t2(u2)
        u4 = self.conv_t3(u3)

        out = self.conv4(u4)
        return out


if __name__ == "__main__":
    model = MTRNet()
