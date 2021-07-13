"""
pipeline.py - Entire Texture Fields pipeline

(Texture Fields: Learning Texture Representations in Function Space, Oechsle et al., ICCV 2019)
"""

import os
import trimesh

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist

from .image_encoder import TextureFieldsImageEncoder
from .shape_encoder import TextureFieldsShapeEncoder
from .texture_field_core import TextureFieldsCore
from .gan_discriminator import TextureFieldsGANDiscriminator
from .vae_encoder import TextureFieldsVAEEncoder


class TextureFieldsCls(nn.Module):
    def __init__(self):
        super(TextureFieldsCls, self).__init__()

    def forward(self, depth, cam_K, cam_R, z, s):
        """
        Forward propagation.

        Args:
        - depth (torch.Tensor): Tensor of shape (B, 1, H, W). Batch of depth maps.
        - cam_K (torch.Tensor): Tensor of shape (B, 3, 4). Batch of camera projection matrices.
        - cam_R (torch.Tensor): Tensor of shape (B, 3, 4). Batch of camera matrices representing camera's translation and rotation.
        - z (torch.Tensor): Tensor of shape (B, c_dim) where c_dim is the dimensionality of image feature vector.
        - s (torch.Tensor): Tensor of shape (B, out_dim) where out_dim is the dimensionality of shape feature vector.

        Returns:
        - Tensor of shape (B, 3, H * W). Batch of RGB color coordinates at points obtained by unprojecting input image pixels.
        """
        batch_size, _, W, H = depth.size()
        assert(depth.size(1) == 1)
        assert(cam_K.size() == (batch_size, 3, 4))
        assert(cam_R.size() == (batch_size, 3, 4))

        coord