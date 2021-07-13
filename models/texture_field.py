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
from .decoder import TextureFieldsDecoder


class TextureFieldsCls(nn.Module):
    def __init__(self, use_VAE=False, p_z=None, white_bg=True, use_MAP=False):
        """
        Constructor of TextureFieldsCls.

        Args:
        - use_vae (bool): Determines whether to use VAE encoder or conditional image encoder.
        - p_z (torch.Distribution): Initial probability distribution for variational inference.
        - white_bg (bool): Indicates whether the background of input images are white or not.
        - use_MAP (bool): Indicates whether to use MAP or sampling for variational inference.
        """
        super(TextureFieldsCls, self).__init__()

        self.use_VAE = use_VAE
        self.white_bg = white_bg
        self.use_MAP = use_MAP

        if self.use_VAE:
            self.encoder = TextureFieldsVAEEncoder()
            if p_z is None:
                self.p_z = dist.Normal(torch.tensor([]), torch.tensor([]))
            else:
                self.p_z = p_z
        else:
            self.encoder = TextureFieldsImageEncoder()
            self.p_z = None

        self.shape_encoder = TextureFieldsShapeEncoder()
        self.decoder = TextureFieldsDecoder()

    def forward(self, depth, cam_K, cam_R, geometry, condition):
        """
        Forward propagation.

        Args:
        - depth (torch.Tensor): Tensor of shape (B, 1, H, W). Batch of depth maps.
        - cam_K (torch.Tensor): Tensor of shape (B, 3, 4). Batch of camera projection matrices.
        - cam_R (torch.Tensor): Tensor of shape (B, 3, 4). Batch of camera matrices representing camera's translation and rotation.
        - geometry (dict): Dictionary containing one or more geometric features of input shapes
                - e.g. { 'points': (torch.Tensor / <B, 3, N>), 'normals': (torch.Tensor / <B, 3, N>)}
        - condition (torch.Tensor): Tensor of shape (B, 3, H, W). Batch of conditional images.

        Returns:
        - img (torch.Tensor): Tensor of shape (B, 3, H, W). 
                Batch of RGB color coordinates at points obtained by unprojecting input image pixels.
        """
        batch_size, _, W, H = depth.size()
        assert depth.size(1) == 1
        assert cam_K.size() == (batch_size, 3, 4)
        assert cam_R.size() == (batch_size, 3, 4)

        coord, mask = self.unproject_depth(depth, cam_K, cam_R)

        # get image latent vector 'z' either by:
        # (1) extracting it from given conditional image
        # (2) variational inference (sampling from distribution or MAP)
        if self.use_VAE:
            z = self.get_z_from_prior((batch_size,))
        else:
            assert condition is not None
            z = self.encoder(condition)
            z = z.cuda()

        # get shape latent vector 's'
        s = self.shape_encoder(geometry)

        # map texture colors in 3D space onto 2D image
        coord = coord.view(batch_size, 3, W * H)
        rgb = self.decoder(coord, z)
        rgb = rgb.view(batch_size, 3, W, H)

        if self.white_bg:
            rgb_bg = torch.ones_like(rgb)
        else:
            rgb_bg = torch.zeros_like(rgb)

        img = (mask * rgb).permute(0, 1, 3, 2) + (1 - mask.permute(0, 1, 3, 2)) * rgb_bg

        return img

    # helper functions
    def unproject_depth(self, depth, cam_K, cam_R):
        """
        Perform unprojection on given depth map.

        Args:
        - depth (torch.Tensor): Tensor of shape (B, 1, H, W). Batch of depth maps.
        - cam_K (torch.Tensor): Tensor of shape (B, 3, 4). Batch of camera projection matrices.
        - cam_R (torch.Tensor): Tensor of shape (B, 3, 4). Batch of camera matrices representing camera's translation and rotation.

        Returns:
        - coord (torch.Tensor): Tensor of shape (B, 3, H, W). 
                Coordinates obtained by unprojecting given images with camera parameters and extrinsic.
        - mask (torch.Tensor): Tensor of shape (B, 1, H, W). Mask indicating the pixels with finite depth values.
        """

        assert depth.size(1) == 1
        batch_size, _, H, W = depth.size()
        device = depth.device

        depth = torch.permute(depth, (0, 1, 3, 2))  # (B, 1, H, W) -> (B, 1, W, H)
        depth = -depth  # reverse the sign of depth (Negative in OpenGL by default)

        affine_fourth_row = torch.tensor([0.0, 0.0, 0.0, 1.0])
        affine_fourth_row = affine_fourth_row.expand((batch_size, 1, 4)).to(device)

        # add fourth row to camera (extrinsic) matrices
        cam_R = torch.cat((cam_R, affine_fourth_row), dim=1)

        # build mask indicating pixels with valid depth values (i.e. mapped from object, not background)
        mask = (~torch.isinf(depth)).float()
        depth[torch.isinf(depth)] = 0

        d = depth.reshape(batch_size, 1, W * H)

        # create tensor for pixel locations
        pixel_x, pixel_y = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        pixel_x, pixel_y = pixel_x.to(device), pixel_y.to(device)
        pixels = torch.cat(
            (
                pixel_x.expand(batch_size, 1, pixel_x.size(0), pixel_x.size(1)),
                (W - pixel_y).expand(
                    batch_size, 1, pixel_y.size(0), pixel_y.size(1)
                ),  # reverse the index for height
            ),
            dim=1,
        )
        pixels = pixels.reshape(batch_size, 2, pixel_y.size(0) * pixel_y.size(1))
        pixels = pixels.float() / W * 2

        # create terms of mapping equation x = P^-1 * d*(qp - b)
        P = cam_K[:, :2, :2].float().to(device)
        q = cam_K[:, 2:3, 2:3].float().to(device)
        b = cam_K[:, :2, 2:3].expand(batch_size, 2, d.size(2)).to(device)
        inv_P = torch.inverse(P).to(device)

        operand = (pixels.float() * q.float() - b.float()) * d.float()
        x_xy = torch.bmm(inv_P, operand)

        x_world = torch.cat((x_xy, d, torch.ones_like(d)), dim=1)

        inv_R = torch.inverse(cam_R)
        coord = torch.bmm(inv_R.expand(batch_size, 4, 4), x_world).reshape(batch_size, 4, H, W)

        coord = coord[:, :3].to(device)
        mask = mask.to(device)

        return coord, mask

    def get_z_from_prior(self, size):
        """
        Draw latent code z from prior either by sampling distribution or MAP.

        Args:
        - size (torch.Size): Size of sample to draw.

        Returns:
        -  z (torch.Tensor): Tensor of shape (*size, z.size). Latent code.
        """

        if self.use_MAP:
            z = self.p_z.mean
            z = z.expand(*size, *z.size())
        else:
            z = self.p_z.sample(size)

        return z

