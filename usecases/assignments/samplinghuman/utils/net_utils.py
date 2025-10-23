# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Network utility functions"""

import numpy as np
import torch
from torch import nn


def initialize_weights(layer: nn.Module):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


def sinusoidal_positional_encoding(d_model, tile_size):
    d_model = d_model // 4  # Since we have 2D positional encoding, we reduce the dimension accordingly
    position = np.arange(tile_size * tile_size)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    row_pos = np.repeat(np.arange(0, tile_size), tile_size)[:, np.newaxis]
    col_pos = np.tile(np.arange(0, tile_size), tile_size)[:, np.newaxis]

    pe = np.zeros((tile_size * tile_size, d_model))
    pe[:, 0::2] = np.sin(position * div_term) * np.sin((row_pos / tile_size) * 2 * np.pi) * np.cos((col_pos / tile_size) * 2 * np.pi)
    pe[:, 1::2] = np.cos(position * div_term) * np.sin((row_pos / tile_size) * 2 * np.pi) * np.cos((col_pos / tile_size) * 2 * np.pi)

    return torch.tensor(pe, dtype=torch.float32).flatten()
