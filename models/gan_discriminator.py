"""
gan_discriminator.py - 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_block import TextureFieldsResNetBlock


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
        self.maxpool = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))  # (4, 4) -> (1, 1)
        self.fc_out = nn.Linear(512, 1)

    def forward(self, rgb, depth):
        """
        Forward propagation.

        Args:
        - rgb (torch.Tensor): Tensor of shape (B, 3, H, W). 2D image.
        - depth (torch.Tensor): Tensor of shape (B, 1, H, W). 2D depth map.

        Returns:
        - Tensor of shape (B,). Tensor of probability that images in the batch are fake.
        """

        depth = depth.clone()
        depth[torch.isinf(depth)] = 0

        x = torch.cat((rgb, depth), dim=1)

        x = F.relu(self.conv_in(x))

        for i in range(len(self.resnet_blocks)):
            x = self.resnet_blocks[i](x)

        x = self.maxpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc_out(x)

        return F.sigmoid(x)
