"""
resnet_block.py - ResNet block module for Texture Fields components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureFieldsResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        """
        Constructor of TextureFieldsVAEEncoderResNetBlock.

        Args:
        - in_channel (int): Number of channels of input feature map. Then its shape becomes (B, in_channel, H, W)
        - out_channel (int): Number of channels of output feature map. Then its shape becomes (B, out_channel, H, W)
        """
        super(TextureFieldsResNetBlock, self).__init__()

        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=(1, 1), stride=(1, 1))
        self.conv_2 = nn.Conv2d(in_channel, in_channel, kernel_size=(1, 1), stride=(1, 1))
        self.conv_3 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        """
        Forward propagation.

        Args:
        - x (torch.Tensor): Tensor of shape (B, in_channel, H, W).

        Returns:
        - Tensor of shape (B, out_channel, H, W). Output feature map.
        """

        skip = x.clone()
        x = F.relu(self.conv_1(x))
        x = self.conv_2(x)
        x = F.relu(x + skip)
        x = F.relu(self.conv_3(x))
        x = self.maxpool(x)

        return x
