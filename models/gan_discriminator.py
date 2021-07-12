"""
gan_discriminator.py - 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_block import TextureFieldsResNetBlock

# TODO: Finish implementing GAN discriminator!


class TextureFieldsGANDiscriminator(nn.Module):
    def __init__(self):
        """

        """
        super(TextureFieldsGANDiscriminator, self).__init__()

        self.conv_in = nn.Conv2d(4, 32, kernel_size=(1, 1), stride=(1, 1))
        self.resnet_blocks = nn.ModuleList(
            [
                TextureFieldsResNetBlock(32, 64),  # (128, 128) -> (64, 64)
                TextureFieldsResNetBlock(64, 128),  # (64, 64) -> (32, 32)
                TextureFieldsResNetBlock(128, 128),  # (32, 32) -> (16, 16)
                TextureFieldsResNetBlock(128, 256),  # (16, 16) -> (8, 8)
                TextureFieldsResNetBlock(256, 512),  # (8, 8) -> (4, 4)
            ]
        )
        self.maxpool = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))

    def forward(self, x):
        """
        Forward propagation.

        Args:
        - x (torch.Tensor): Tensor of shape (B, 4, H, W). 2D image but width additional channel for depth

        Returns:
        - Tensor of shape (B,). Tensor of probability that images in the batch are fake.
        """

        x = F.relu(self.conv_in(x))

        for i in range(len(self.resnet_blocks)):
            x = self.resnet_blocks[i](x)

        x = self.maxpool(x)

        return F.sigmoid(x)
