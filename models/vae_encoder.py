"""
vae_encoder.py - VAE encoder network for Texture Fields architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_block import TextureFieldsResNetBlock


class TextureFieldsVAEEncoder(nn.Module):
    def __init__(self, s_dim=512):
        """
        Constructor of TextureFieldsVAEEncoder.

        Args:
        - s_dim (int): Dimensionality of shape feature vector (or tensor). Set to 512 by default.
        """
        super(TextureFieldsVAEEncoder, self).__init__()

        self.fc_feature = nn.Linear(s_dim, 32)

        self.conv_in = nn.Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1))

        self.resnet_blocks = nn.ModuleList(
            [
                TextureFieldsResNetBlock(32, 64),  # (128, 128)
                TextureFieldsResNetBlock(64, 128),  # (64, 64)
                TextureFieldsResNetBlock(128, 128),  # (32, 32)
                TextureFieldsResNetBlock(128, 256),  # (16, 16)
                TextureFieldsResNetBlock(256, 512),  # (8, 8)
            ]
        )

        # Map (512, 4, 4) feature map to feature vector (512,)...? How?
        self.fc_mu = nn.Linear(512 * 4 * 4, 512)
        self.fc_logvar = nn.Linear(512 * 4 * 4, 512)

    def forward(self, x, s):
        """
        Forward propagation.

        Args:
        - x (torch.Tensor): Tensor of shape (B, 3, H, W). Batch of 2D images.
        - s (torch.Tensor): Tensor of shape (B, s_dim)

        Returns:
        - Tuple of tensors of shape (B, 512). Each of which is mean and variance for image distribution, respectively.
        """
        x = self.conv_in(x)  # (B, 32, 128, 128)

        s = self.fc_feature(s)  # (B, 32)
        s = s.unsqueeze(2)
        s = s.unsqueeze(3)
        s = s.repeat(1, 1, 128, 128)

        # add shape feature to image feature
        x += s

        # go through series of ResNet blocks
        for i in range(len(self.resnet_blocks)):
            x = self.resnet_blocks[i](x)

        x_shape = x.size()
        x = x.view(x_shape[0], -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar
