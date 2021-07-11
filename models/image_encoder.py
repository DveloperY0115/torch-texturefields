
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImageEncoder(nn.Module):

    def __init__(self, pretrained=True):
        self.model = models.resnet18(pretrained=True)

    def forward(self, x):
        return self.model(x)
        