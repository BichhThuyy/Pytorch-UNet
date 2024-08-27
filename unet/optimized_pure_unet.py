import torch.nn as nn

from unet.unet_parts import DoubleConv, Down, Up, OutConv
from unet.attention_module import SpatialAttention


class OptimisedUNetWithSpatialAttention(nn.Module):
    def __init__(self, in_channels, n_classes, bilinear=True):
        super(OptimisedUNetWithSpatialAttention, self).__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 96)
        self.down2 = Down(96, 128)
        self.down3 = Down(128, 192)
        self.down4 = Down(192, 256)
        self.down5 = Down(256, 384)
        self.down6 = Down(384, 512)
        self.down7 = Down(512, 512)

        self.attention = SpatialAttention()

        self.up1 = Up(1024, 384, bilinear)
        self.up2 = Up(768, 256, bilinear)
        self.up3 = Up(512, 192, bilinear)
        self.up4 = Up(384, 128, bilinear)
        self.up5 = Up(256, 96, bilinear)
        self.up6 = Up(192, 64, bilinear)
        self.up7 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)

        # Add Attention module here
        x8 = self.attention(x8)

        x = self.up1(x8, x7)
        x = self.up2(x, x6)
        x = self.up3(x, x5)
        x = self.up4(x, x4)
        x = self.up5(x, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)
        logits = self.outc(x)
        return logits
