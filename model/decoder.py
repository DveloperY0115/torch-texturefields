"""
decoder.py - Decoder network for mapping color field in 3D to 2D image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_block import TextureFieldsResNetBlockPointwise


class TextureFieldsDecoder(nn.Module):
    def __init__(self):
        """
        Constructor of TextureFieldsDecoder.

        Args:
        - 
        """
        super(TextureFieldsDecoder, self).__init__()

    def forward(self, *args, **kwargs):
        """
        Forward propagation.

        Args:
        - 
        """
        raise NotImplementedError
