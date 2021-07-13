"""
image_encoder.py - Subnetwork for encoding images in conditional setup.

Pretrained ResNet-18 is used.
"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class TextureFieldsImageEncoder(nn.Module):
    def __init__(self, c_dim=128, normalize=True, use_linear=True):
        """
        Constructor of TextureFieldsImageEncoder.

        Fetches pretrained ResNet-18 and apply modification.

        Args:
        - c_dim (int): Dimensionality of output feature vector.
        - normalize (bool): Indicator for input normalization.
        - use_linear (bool): Indicator for using fully connected layer as output layer.
        """
        super(TextureFieldsImageEncoder, self).__init__()

        self.normalize = normalize
        self.use_linear = use_linear
        self.model = models.resnet18(pretrained=True, progress=True)
        self.model.fc = nn.Sequential()  # ResNet-18 originally outputs score for 1000 classes

        # freeze parameters of the network
        for param in self.parameters():
            param.requires_grad = False

        if use_linear:
            self.model.fc = nn.Linear(512, c_dim)
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif c_dim == 512:
            pass  # do nothing
        else:
            raise ValueError("c_dim must be 512 if use_linear is False")

    def forward(self, x):
        """
        Forward propagation.

        Args:
        - x (torch.Tensor): A batch of images of shape (B, 3, W, H)

        Returns:
        - A feature tensor of shape (B, c_dim)
        """

        if self.normalize:
            x = normalize_imagenet(x)
        return self.model(x)


def normalize_imagenet(x):
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x
