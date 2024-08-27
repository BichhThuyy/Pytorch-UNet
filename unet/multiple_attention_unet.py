import torch
import torch.nn as nn

from unet.unet_parts import DoubleConv, Down, Up, OutConv
from unet.attention_module import SpatialAttention


class UNetWithMultipleSpatialAttention(nn.Module):
    def __init__(self, in_channels, n_classes, bilinear=True):
        super(UNetWithMultipleSpatialAttention, self).__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.attention = SpatialAttention()

        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.attention(x2)

        x3 = self.down2(x2)
        x3 = self.attention(x3)

        x4 = self.down3(x3)
        x4 = self.attention(x4)

        x5 = self.down4(x4)
        x5 = self.attention(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
