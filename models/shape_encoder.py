"""
shape_encoder.py - Subnetwork for encoding point clouds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureFieldsShapeEncoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=512, L=4):
        """
        Constructor of TextureFieldsShapeEncoder.

        Args:
        - L (int): Number of ResNet-like blocks. Set to 4 by default.
        """
        super(TextureFieldsShapeEncoder, self).__init__()

        self.L = L

        self.fc_1 = nn.Conv1d(in_dim, 256, 1)
        self.fc_2 = nn.Conv1d(256, 128, 1)
        self.fc_3 = nn.Conv1d(128, 128, 1)
        self.fc_4 = nn.Conv1d(128, out_dim, 1)
        self.resnet_blocks = nn.ModuleList([PointNetResNetBlock() for _ in range(self.L)])

    def forward(self, x):
        """
        Forward propagation.

        Args:
        - x (torch.Tensor): A tensor of shape (B, N, 3), where 
            - B: size of a batch
            - N: number of points in a point cloud
       
        Returns:
        - x (torch.Tensor): Tensor of shape (B, out_dim). Features of input point cloud.
        """

        x = x.transpose(1, 2)

        x = F.relu(self.fc_1(x))

        for i in range(self.L):
            x = F.relu(self.resnet_blocks[i](x))

        x = F.relu(self.fc_2(x))
        skip = x.clone()
        x = F.relu(self.fc_3(x))
        x += skip

        x, _ = torch.max(x, dim=2, keepdim=True)
        x = self.fc_4(x)

        return x.squeeze()


class PointNetResNetBlock(nn.Module):
    def __init__(self, hidden_dim=128):
        """
        Constructor of PointNetResNetBlock.

        Args:
        - hidden_dim (int): Dimensionality of hidden feature. Set to 128 by default.
        """
        super(PointNetResNetBlock, self).__init__()

        self.fc_1 = nn.Conv1d(2 * hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, x):
        """
        Forward propagation.

        Args:
        - x (torch.Tensor): Tensor of shape (N, 2 * hidden_dim, 1)
        
        Returns:
        - A tensor of shape (N, 2 * hidden_dim, 1)
        """

        # feed-forward and residual connection
        x = F.relu(self.fc_1(x))
        skip = x.clone()
        x = F.relu(self.fc_2(x))
        x += skip

        # max pooling, expand and concatenate
        num_points = x.size()[0]
        skip = x.clone()
        x, _ = torch.max(x, dim=0, keepdim=True)
        x = x.repeat(num_points, 1, 1)
        x = torch.cat((x, skip), dim=1)

        return x
