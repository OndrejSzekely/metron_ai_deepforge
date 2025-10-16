# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Image VAE Model"""

from torch import nn
from torch.nn import functional as F


def conv_block(in_channels, out_channels, kernel_size, stride, padding, use_bn=True):
    """Convolutional Block with optional BatchNorm and ReLU."""
    layers: list[nn.Conv2d | nn.BatchNorm2d] = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class VAEEncoder(nn.Module):
    """Based on ResNet18

    Input resolution is 16x16x3

    Architecture:
    - Conv2D 3x3, stride 1, padding 1, out_channels 64 + BN + ReLU => 16x16x64
    - Conv2D 3x3, stride 1, padding 1, out_channels 64 + BN + ReLU => 16x16x64
    - MaxPool 2x2, stride 2 => 8x8x64
    - Conv2D 3x3, stride 1, padding 1, out_channels 64 + BN + ReLU => 8x8x64
    - Conv2D 3x3, stride 1, padding 1, out_channels 64 + BN => 8x8x64
    - Residual connection + ReLU => 8x8x64
    - MaxPool 2x2, stride 2 => 4x4x64
    - Conv2D 3x3, stride 1, padding 1, out_channels 128 + BN + ReLU => 4x4x128
    - Conv2D 3x3, stride 1, padding 1, out_channels 128 + BN => 4x4x128
    - Residual connection projected to 128 + ReLU => 4x4x128
    - MaxPool 2x2, stride 2 => 2x2x256
    - Conv2D 3x3, stride 1, padding 1, out_channels 256 + BN + ReLU => 2x2x256
    - Conv2D 3x3, stride 1, padding 1, out_channels 256 + BN => 2x2x256
    - Residual connection projected to 256 + ReLU => 2x2x256
    - MaxPool 2x2, stride 2 => 1x1x256
    - Flatten => 256
    - Linear 256 -> 256 + ReLU => 256
    """

    def __init__(self):
        super().__init__()
        self.conv_block1 = conv_block(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding="same")
        self.conv_block2 = conv_block(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same")
        self.conv_block3 = conv_block(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same")
        self.conv_block4 = conv_block(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same")
        self.conv_block5 = conv_block(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding="same")
        self.conv_block6 = conv_block(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding="same")
        self.conv_block7 = conv_block(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding="same")
        self.conv_block8 = conv_block(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding="same")
        self.fc1 = nn.Linear(256, 256)
        self.point_conv1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.point_conv2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): Tensor of shape (B, C, H, W)
        """
        x = F.relu(self.conv_block1(x))
        x = F.relu(self.conv_block2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        residual = x
        x = F.relu(self.conv_block3(x))
        x = self.conv_block4(x)
        x += residual
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        residual = x
        x = F.relu(self.conv_block5(x))
        x = self.conv_block6(x)
        x = x + self.point_conv1(residual)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        residual = x
        x = F.relu(self.conv_block7(x))
        x = self.conv_block8(x)
        x = x + self.point_conv2(residual)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)

        return x
