"""
shape_encoder.py - Subnetwork for encoding point clouds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureFieldsShapeEncoder(nn.Module):
    def __init__(self, in_dim=6, out_dim=512, hidden_dim=128, L=4):
        """
        Constructor of TextureFieldsShapeEncoder.

        Args:
        - in_dim (int): Dimensionality of input point cloud. Set to 6 by default.
        - out_dim (int): Dimensionality of output per-point feature vector. Set to 512 by default.
        - hidden_dim (int): Dimensionality of hidden feature vector within this ResNet block. Set to 128 by default.
        - L (int): Number of ResNet-like blocks. Set to 4 by default.
        """
        super(TextureFieldsShapeEncoder, self).__init__()

        self.L = L

        self.fc_1 = nn.Conv1d(in_dim, 2 * hidden_dim, 1)
        self.fc_2 = nn.Conv1d(2 * hidden_dim, hidden_dim, 1)
        self.fc_3 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_4 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.resnet_blocks = nn.ModuleList([PointNetResNetBlock() for _ in range(self.L)])

    def forward(self, x):
        """
        Forward propagation.

        Args:
        - x (torch.Tensor): A tensor of shape (B, in_dim, N), where 
            - B: size of a batch
            - N: number of points in a point cloud
       
        Returns:
        - x (torch.Tensor): Tensor of shape (B, out_dim). Features of input point cloud.
        """
        x = F.relu(self.fc_1(x))

        for i in range(self.L):
            x = F.relu(self.resnet_blocks[i](x))

        x = F.relu(self.fc_2(x))
        skip = x.clone()
        x = F.relu(self.fc_3(x))
        x += skip

        x, _ = torch.max(x, dim=2, keepdim=True)
        x = self.fc_4(x)

        x = x.squeeze()

        if len(x.size()) == 1:
            x = x.unsqueeze(0)

        return x


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
        self.short_cut = nn.Conv1d(2 * hidden_dim, hidden_dim, 1, bias=False)

        # initialization
        nn.init.zeros_(self.fc_2.weight)

    def forward(self, x):
        """
        Forward propagation.

        Args:
        - x (torch.Tensor): Tensor of shape (N, 2 * hidden_dim, 1)
        
        Returns:
        - A tensor of shape (N, 2 * hidden_dim, 1)
        """

        # feed-forward and residual connection
        skip = self.short_cut(x)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x += skip

        # max pooling, expand and concatenate
        num_points = x.size()[2]
        skip = x.clone()
        x, _ = torch.max(x, dim=2, keepdim=True)
        x = x.repeat(1, 1, num_points)
        x = torch.cat((x, skip), dim=1)

        return x
