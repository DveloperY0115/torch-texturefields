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
    def __init__(self, mode, device, white_bg=True, use_MAP=False):
        """
        Constructor of TextureFieldsCls.

        Args:
        - mode (str): String indicates which mode to use. Can be one of 'conditional', 'vae', and 'gan'.
        - device (torch.device): Object representing currently used device.
        - white_bg (bool): Indicates whether the background of input images are white or not.
        - use_MAP (bool): Indicates whether to use MAP or sampling for variational inference.
        """
        super(TextureFieldsCls, self).__init__()

        self.mode = mode
        self.device = device
        self.white_bg = white_bg
        self.use_MAP = use_MAP

        # initialize posterior distribution
        if self.mode == "vae":
            self.q_z = None

        # set experimental setting specific submodules
        if self.mode == "conditional":
            self.encoder = TextureFieldsImageEncoder(c_dim=512)
        else:
            if self.mode == "vae":
                self.encoder = TextureFieldsVAEEncoder()
            else:
                # GAN
                self.encoder = None

        # initialize shape encoder
        self.shape_encoder = TextureFieldsShapeEncoder()

        # initialize Texture Field core
        self.core = TextureFieldsCore(z_dim=512)

    def forward(self, depth, cam_K, cam_R, geometry, condition, real):
        """
        Forward propagation.

        Args:
        - depth (torch.Tensor): Tensor of shape (B, 1, H, W). Batch of depth maps.
        - cam_K (torch.Tensor): Tensor of shape (B, 3, 4). Batch of camera projection matrices.
        - cam_R (torch.Tensor): Tensor of shape (B, 3, 4). Batch of camera matrices representing camera's translation and rotation.
        - geometry (dict): Dictionary containing one or more geometric features of input shapes
                - e.g. { 'points': (torch.Tensor / <B, 3, N>), 'normals': (torch.Tensor / <B, 3, N>)}
        - condition (torch.Tensor): Tensor of shape (B, 3, H, W). Batch of conditional images.
        - real (torch.Tensor): Tensor of shape (B, 3, H, W). Batch of ground truth images.

        Returns:
        - out (Dict): Dictionary containing outputs of model such as predicted image, loss, etc..
            - out["img_pred"] (torch.Tensor): Tensor of shape (B, 3, H, W). Batch of predicted images
            - out["loss"] (scalar): Computed loss. Can be used for back propagation later.
            NOTE: For GAN training, this method returns only 
        """
        batch_size, _, W, H = depth.size()
        assert depth.size(1) == 1
        assert cam_K.size() == (batch_size, 3, 4)
        assert cam_R.size() == (batch_size, 3, 4)

        coord, mask = self.unproject_depth(depth, cam_K, cam_R)

        coord = coord.reshape(batch_size, 3, W * H)  # (B, 3, H, W) -> (B, 3, H * W)
        coord = coord.transpose(1, 2)  # (B, 3, H * W) -> (B, H * W, 3)

        # get shape latent vector 's'
        s = self.shape_encoder(geometry)

        # get image latent vector 'z'
        if self.mode == "gan":
            z = self.get_z_from_prior((batch_size,))
        else:
            if self.mode == "conditional":
                assert (
                    condition is not None
                ), "[!] Condition image must be provided in conditional setting!"
                z = self.encoder(condition)
            else:  # VAE
                mu, logvar = self.encoder(real, s)
                self.q_z = dist.Normal(mu, torch.exp(logvar))  # update posterior
                z = self.q_z.rsample()

        # infer color for each query points
        rgb = self.core(coord, z, s)
        rgb = rgb.view(batch_size, 3, W, H)

        if self.white_bg:
            rgb_bg = torch.ones_like(rgb)
        else:
            rgb_bg = torch.zeros_like(rgb)

        # dictionary of outputs
        out = {}

        img = (mask * rgb).permute(0, 1, 3, 2) + (1 - mask.permute(0, 1, 3, 2)) * rgb_bg

        out["img_pred"] = img

        # compute experiment specific losses
        if self.mode == "conditional" or self.mode == "gan":
            out["loss"] = nn.L1Loss()(img, real)
        else:  # VAE
            p0_z = dist.Normal(torch.zeros(512).to(self.device), torch.ones(512).to(self.device))
            reconstruction_loss = F.mse_loss(img, real).sum(dim=-1).mean()
            kl = dist.kl_divergence(self.q_z, p0_z).sum(dim=-1).mean() / float(H * W * 3)
            out["loss"] = reconstruction_loss + kl  # ELBO

        return out

    # helper functions
    def generate_images(self, depth, cam_K, cam_R, geometry):
        """
        Generate images given coordinates in 3D space and shape latent code.
        Used for image generation at test time.

        Args:
        - depth (torch.Tensor): Tensor of shape (B, 1, H, W). Batch of depth maps.
        - cam_K (torch.Tensor): Tensor of shape (B, 3, 4). Batch of camera projection matrices.
        - cam_R (torch.Tensor): Tensor of shape (B, 3, 4). Batch of camera matrices representing camera's translation and rotation.
        - geometry (dict): Dictionary containing one or more geometric features of input shapes
                - e.g. { 'points': (torch.Tensor / <B, 3, N>), 'normals': (torch.Tensor / <B, 3, N>)}

        Returns:
        - img (torch.Tensor): Tensor of shape (B, 3, H, W). Batch of generated images.
        """
        with torch.no_grad():
            assert (
                self.mode != "conditional"
            ), "[!] Conditional model cannot be used for image generation!"

            batch_size, _, W, H = depth.size()
            assert depth.size(1) == 1
            assert cam_K.size() == (batch_size, 3, 4)
            assert cam_R.size() == (batch_size, 3, 4)

            coord, mask = self.unproject_depth(depth, cam_K, cam_R)

            coord = coord.reshape(batch_size, 3, W * H)  # (B, 3, H, W) -> (B, 3, H * W)
            coord = coord.transpose(1, 2)  # (B, 3, H * W) -> (B, H * W, 3)

            # get shape latent vector 's'
            s = self.shape_encoder(geometry)

            if self.mode == "vae":
                z = self.get_z_from_posterior()
            else:  # GAN
                z = self.get_z_from_prior()

            rgb = self.core(coord, z, s)
            rgb = rgb.view(batch_size, 3, W, H)

            if self.white_bg:
                rgb_bg = torch.ones_like(rgb)
            else:
                rgb_bg = torch.zeros_like(rgb)

            img = (mask * rgb).permute(0, 1, 3, 2) + (1 - mask.permute(0, 1, 3, 2)) * rgb_bg

            return img

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

    def get_z_from_prior(self):
        """
        Draw latent code z from prior (standard normal) either by sampling or MAP.

        Args:
        - size (torch.Size): Size of sample to draw.

        Returns:
        -  z (torch.Tensor): Tensor of shape (*size, z.size). Latent code.
        """
        p0_z = dist.Normal(torch.zeros(512).to(self._device), torch.ones(512).to(self._device))

        if self.use_MAP:
            z = p0_z.mean
        else:
            z = p0_z.sample()
        return z

    def get_z_from_posterior(self, use_reparametrization=False):
        """
        Sample image latent vector z from posterior distribution q_z.

        Args:
        - is_differentiable (bool): Determines whether to use reparametrization trick during sampling.

        Returns:
        - z (torch.Tensor): Tensor of shape (B, z_dim). Batch of image latent vectors.
        """
        assert self.q_z is not None, "[!] Posterior must be set in advance for sampling"
        if use_reparametrization:
            z = self.q_z.rsample()
        else:
            z = self.q_z.sample()
        return z

