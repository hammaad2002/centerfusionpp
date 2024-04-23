import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d, deform_conv2d

class DCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, 
                 deformable_groups):
        super(DCN, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size[0] * kernel_size[0],
                                     kernel_size=kernel_size[0], stride=stride, padding=padding, dilation=dilation,
                                     groups=deformable_groups)
        self.deform_conv = deform_conv2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        # self.layerOne = DeformConv2d(in_channels=in_channels, out_channels=out_channels,
        #                              kernel_size=kernel_size[0], stride=stride, padding=padding,
        #                              dilation=dilation, groups=deformable_groups)

    def forward(self, x):
        offset = self.offset_conv(x)
        output = self.deform_conv(x, offset=offset, weight=self.weight, stride=1, padding=1, dilation=1)
        # output = self.layerOne(x)
        return output