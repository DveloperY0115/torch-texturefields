"""
texture_fields.py - The core of Texture Fields architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureFieldsCore(nn.Module):
    def __init__(self, z_dim=512, s_dim=512, L=6):
        """
        Constructor of TextureFieldsCore.

        Args:
        - z_dim (int): Dimensionality of image feature vector (or tensor). Set to 512 by default.
        - s_dim (int): Dimensionality of shape feature vector (or tensor). Set to 512 by default.
        - L (int): Number of ResNet-like blocks in the pipeline. Set to 6 by default.
        """
        super(TextureFieldsCore, self).__init__()

        self.z_dim = z_dim
        self.s_dim = s_dim
        self.L = L

        self.fc_1 = nn.Conv1d(3, 128, 1)
        self.fc_2 = nn.Conv1d(128, 3, 1)
        self.resnet_blocks = nn.ModuleList(
            [TextureFieldsCoreResNetBlock(z_dim=z_dim, s_dim=s_dim) for _ in range(self.L)]
        )

    def forward(self, p, z, s):
        """
        Forward propagation.

        Args:
        - p (torch.Tensor): Tensor of shape (B, N, 3). 3-coordinates of points 
        - z (torch.Tensor): Tensor of shape (B, img_feature_dim). Image latent code from image encoder
        - s (torch.Tensor): Tensor of shape (B, shape_feature_dim). Shape latent code from shape encoder 

        Returns:
        - x (torch.Tensor): Tensor of shape (B, 3). Predicted color vector per point in 'p'
        """

        # concatenate (image, shape) features
        features = torch.cat((z, s), dim=1)

        # obtain per-point feature from input point cloud
        if p.size()[1] != 3:
            p = p.transpose(1, 2)  # (B, N, 3) -> (B, 3, N)
        x = self.fc_1(p)

        for i in range(self.L):
            x = F.relu(self.resnet_blocks[i](features, x))

        x = self.fc_2(x)
        x = torch.sigmoid(x)
        return x


class TextureFieldsCoreResNetBlock(nn.Module):
    def __init__(self, z_dim=512, s_dim=512, hidden_dim=128):
        """
        Constructor of TextureFieldsCoreResNetBlock.

        Args:
        - z_dim (torch.Tensor): Dimensionality of image feature vector (or tensor). Set to 128 by default.
        - s_dim (torch.Tensor): Dimensionality of shape feature vector (or tensor). Set to 512 by default.
        - hidden_dim (torch.Tensor): Dimensionality of hidden feature vector (or tensor) within this block. Set to 128 by default.
        """
        super(TextureFieldsCoreResNetBlock, self).__init__()

        self.fc_1 = nn.Linear(z_dim + s_dim, hidden_dim)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_3 = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, features, points):
        """
        Forward propagation.

        Args:
        - features (torch.Tensor): Tensor of shape (B, img_feature_dim+shape_feature_dim), where B is batch size.
        - points (torch.Tensor): Tensor of shape (B, hidden_dim, N), where
                                - B: Batch size
                                - N: Number of points in input point cloud
        Returns:
        - x (torch.Tensor): Tensor of shape (B, hidden_dim). Tensor of features. 
        """

        features = F.relu(self.fc_1(features))
        features = features.unsqueeze(2)  # (B, hidden_dim) -> (B, hidden_dim, 1)
        features = features.repeat(
            1, 1, points.size()[2]
        )  # (B, hidden_dim, 1) -> (B, hidden_dim, N)
        x = points + features

        skip = x.clone()

        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        x += skip

        return x
