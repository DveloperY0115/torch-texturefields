"""
shape_encoder.py - Subnetwork for encoding point clouds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureFieldsShapeEncoder(nn.Module):
    def __init__(self):
        """
        Constructor of TextureFieldsShapeEncoder.
        """
        super(TextureFieldsShapeEncoder, self).__init__()

        self.conv_256 = nn.Conv1d(3, 256)

    def forward(self, x):
        """
        Forward propagation.

        Args:
        - x (torch.Tensor): A tensor of shape (N, 3) where N is the number of points in a point cloud.
        """


class PointNetResNetBlock(nn.Module):
    def __init__(self, hidden_dim=128):
        """
        Constructor of PointNetResNetBlock.

        Args:
        - hidden_dim (int): Dimensionality of hidden feature. Usually set to 128.
        """
        super(PointNetResNetBlock, self).__init__()

        self.conv_1 = nn.Conv1d(2 * hidden_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, x):
        """
        Forward propagation.

        Args:
        - x (torch.Tensor): A tensor of shape (N, 2 * hidden_dim, 1)
        
        Returns:
        - A tensor of shape (N, 2 * hidden_dim, 1)
        """

        # feed-forward and residual connection
        x = F.relu(self.conv_1(x))
        skip = x.clone()
        x = F.relu(self.conv_2(x))
        x += skip

        # max pooling, expand and concatenate
        num_points = x.size()[0]
        skip = x.clone()
        x, _ = torch.max(x, dim=0)
        x = x.unsqueeze(dim=0)
        x = x.repeat(num_points, 1, 1)
        x = torch.cat((x, skip), dim=1)

        return x
