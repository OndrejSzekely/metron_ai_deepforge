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

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
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
        self.point_conv2 = nn.Conv2d(128, self.embedding_dim, kernel_size=1, stride=1, padding=0)

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
        x = F.relu(x + residual)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        residual = x
        x = F.relu(self.conv_block5(x))
        x = self.conv_block6(x)
        x = F.relu(x + self.point_conv1(residual))
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        residual = x
        x = F.relu(self.conv_block7(x))
        x = self.conv_block8(x)
        x = F.relu(x + self.point_conv2(residual))
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)

        return x


def deconv_block(in_channels, out_channels, kernel_size, stride, padding, output_padding, use_bn=True):
    """Convolutional Block with optional BatchNorm and ReLU."""
    layers: list[nn.ConvTranspose2d | nn.BatchNorm2d] = [
        nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
    ]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class VAEDecoder(nn.Module):
    """Inspired by U-Net Decoder & ResNet

    Input resolution is 256 embedding vector with added positional encoding

    Architecture:
    - Reshape 256 -> 1x1x256
    - ConvTranspose2D 3x3, stride 1, padding 1, out_channels 256 + BN + ReLU  => 2x2x256
    - Conv2D 3x3, stride 1, padding 1, out_channels 256 + BN + ReLU => 2x2x256
    - ConvTranspose2D 3x3, stride 2, padding 1, out_channels 256 + BN + ReLU => 4x4x256
    - Conv2D 3x3, stride 1, padding 1, out_channels 128 + BN => 4x4x128
    - Residual connection projected to 128 + ReLU => 4x4x128
    - ConvTranspose2D 3x3, stride 2, padding 1, out_channels 128 + BN + ReLU => 8x8x128
    - Conv2D 3x3, stride 1, padding 1, out_channels 64 + BN => 8x8x64
    - Residual connection projected to 64 + ReLU => 8x8x64
    -x ConvTranspose2D 3x3, stride 2, padding 1, out_channels 64 + BN + ReLU => 16x16x64
    - Conv2D 3x3, stride 1, padding 1, out_channels 32 + BN => 16x16x32
    - Residual connection projected to 32 + ReLU => 16x16x32
    - Conv2D 3x3, stride 1, padding 1, out_channels 3 + Sigmoid => 16x16x3
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.deconv_block1 = deconv_block(in_channels=self.embedding_dim, out_channels=256, kernel_size=2, stride=1, padding=0, output_padding=0)
        self.conv_block1 = conv_block(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding="same")
        self.deconv_block2 = deconv_block(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, output_padding=0)
        self.conv_block2 = conv_block(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding="same")
        self.deconv_block3 = deconv_block(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_block3 = conv_block(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding="same")
        self.deconv_block4 = deconv_block(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_block4 = conv_block(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding="same")
        self.conv_block5 = conv_block(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding="same", use_bn=False)
        self.point_conv1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.point_conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.point_conv3 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): Tensor of shape (B, 256)
        """
        x = x.view(-1, 256, 1, 1)
        x = F.relu(self.deconv_block1(x))
        x = F.relu(self.conv_block1(x))
        x = F.relu(self.deconv_block2(x))
        residual = x
        x = F.relu(self.conv_block2(x))
        x = F.relu(x + self.point_conv1(residual))
        x = F.relu(self.deconv_block3(x))
        residual = x
        x = F.relu(self.conv_block3(x))
        x = F.relu(x + self.point_conv2(residual))
        x = F.relu(self.deconv_block4(x))
        residual = x
        x = F.relu(self.conv_block4(x))
        x = F.relu(x + self.point_conv3(residual))
        x = self.conv_block5(x)
        x = F.sigmoid(x)

        return x


class VAE(nn.Module):
    """Variational Autoencoder combining VAEEncoder and VAEDecoder."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = VAEEncoder(embedding_dim)
        self.decoder = VAEDecoder(embedding_dim)

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): Tensor of shape (B, C, H, W)
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
