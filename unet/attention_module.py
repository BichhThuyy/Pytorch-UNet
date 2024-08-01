import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3) #32
        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1) #

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, 1, keepdim=True)
        x3 = torch.cat((x1, x2), dim=1)
        x4 = torch.sigmoid(self.conv(x3))
        x = x4 * x
        return x


class InterSliceAttention(nn.Module):
    def __init__(self, in_channels):
        super(InterSliceAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, slice_i, slice_ip1, slice_im1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        slice_i = self.conv(slice_i).to(device)
        attention_mask_ip1 = torch.sigmoid(self.conv(slice_ip1).to(device))
        attention_mask_im1 = torch.sigmoid(self.conv(slice_im1).to(device))

        # attention_mask_ip1 = self.conv(slice_ip1).to(device)
        # attention_mask_im1 = self.conv(slice_im1).to(device)

        slice_ip1_attention = slice_i * attention_mask_ip1
        slice_im1_attention = slice_i * attention_mask_im1

        slice_attention = torch.sigmoid((slice_i + slice_ip1_attention + slice_im1_attention))

        # slice_attention = slice_i +  slice_ip1_attention  +  slice_im1_attention
        # slice_attention = (1 / 3) * slice_i + slice_ip1_attention + slice_im1_attention
        return slice_attention