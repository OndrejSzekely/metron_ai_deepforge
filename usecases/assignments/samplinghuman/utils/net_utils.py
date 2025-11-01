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
    if isinstance(layer, (nn.Linear)):
        nn.init.xavier_uniform_(layer.weight)


def sinusoidal_positional_encoding(d_model, cells_num):
    """Applied on (batch, fragments_num, embedding_dim) => embedding_dim = 16 features for 2x2 cells"""
    d_cell = d_model // (cells_num * cells_num)  # Since we have 2D positional encoding, we reduce the dimension accordingly
    position = np.repeat(np.arange(cells_num * cells_num), d_cell)
    div_term = np.tile(np.repeat(np.exp(np.arange(0, d_cell, 2) * -(np.log(10000.0) / d_cell)), 2), cells_num * cells_num)
    row_pos = np.repeat(np.arange(0, cells_num), d_cell * cells_num)
    col_pos = np.tile(np.repeat(np.arange(0, cells_num), d_cell), cells_num)

    pe = np.zeros((d_model))
    pe[0::2] = np.sin(position[0::2] * div_term[0::2])
    pe[1::2] = np.cos(position[1::2] * div_term[1::2])
    pe = pe * np.sin((row_pos / (cells_num - 1))) * np.cos((col_pos / (cells_num - 1)))

    return torch.tensor(pe, dtype=torch.float32)  # must return => (64)
