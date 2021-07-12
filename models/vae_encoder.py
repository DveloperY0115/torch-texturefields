"""
vae_encoder.py - VAE encoder network for Texture Fields architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureFieldsVAEEncoder(nn.Module):
    def __init__(self, shape_feature_dim=512):
        """
        Constructor of TextureFieldsVAEEncoder.

        Args:
        - shape_feature_dim (int): Dimensionality of shape feature vector (or tensor). Set to 512 by default.
        """
        super(TextureFieldsVAEEncoder, self).__init__()

        self.fc_feature = nn.Linear(shape_feature_dim, 32)

        self.conv_in = nn.Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1))

        self.resnet_blocks = nn.ModuleList(
            [
                TextureFieldsVAEEncoderResNetBlock(32, 64),  # (128, 128)
                TextureFieldsVAEEncoderResNetBlock(64, 128),  # (64, 64)
                TextureFieldsVAEEncoderResNetBlock(128, 128),  # (32, 32)
                TextureFieldsVAEEncoderResNetBlock(128, 256),  # (16, 16)
                TextureFieldsVAEEncoderResNetBlock(256, 512),  # (8, 8)
            ]
        )

        # Map (512, 4, 4) feature map to feature vector (512,)...? How?
        self.fc_mu = nn.Linear(512 * 4 * 4, 512)
        self.fc_logvar = nn.Linear(512 * 4 * 4, 512)

    def forward(self, image, shape_features):
        """
        Forward propagation.

        Args:
        - image (torch.Tensor): Tensor of shape (B, 3, H, W)
        - shape_features (torch.Tensor): Tensor of shape (B, shape_feature_dim)

        Returns:
        - Tuple of tensors of shape (B, 512). Each of which is mean and variance for image distribution, respectively.
        """
        x = self.conv_in(image)  # (B, 32, 128, 128)

        shape_features = self.fc_feature(shape_features)  # (B, 32)
        shape_features = shape_features.unsqueeze(2)
        shape_features = shape_features.unsqueeze(3)
        shape_features = shape_features.repeat(1, 1, 128, 128)

        # add shape feature to image feature
        x += shape_features

        # go through series of ResNet blocks
        for i in range(len(self.resnet_blocks)):
            x = self.resnet_blocks[i](x)

        x_shape = x.size()
        x = x.view(x_shape[0], -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class TextureFieldsVAEEncoderResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        """
        Constructor of TextureFieldsVAEEncoderResNetBlock.

        Args:
        - in_channel (int): Number of channels of input feature map. Then its shape becomes (B, in_channel, H, W)
        - out_channel (int): Number of channels of output feature map. Then its shape becomes (B, out_channel, H, W)
        """
        super(TextureFieldsVAEEncoderResNetBlock, self).__init__()

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

