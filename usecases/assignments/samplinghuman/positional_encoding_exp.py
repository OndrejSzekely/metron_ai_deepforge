# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

import matplotlib.pyplot as plt
import numpy as np


def sinusoidal_positional_encoding(pixels, d_model, tile_size):
    position = np.arange(pixels)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    row_pos = np.repeat(np.arange(0, tile_size), tile_size)[:, np.newaxis]
    col_pos = np.tile(np.arange(0, tile_size), tile_size)[:, np.newaxis]

    pe = np.zeros((max_position, d_model))
    pe[:, 0::2] = np.sin(position * div_term) * np.sin((row_pos / tile_size) * 2 * np.pi) * np.cos((col_pos / tile_size) * 2 * np.pi)
    pe[:, 1::2] = np.cos(position * div_term) * np.sin((row_pos / tile_size) * 2 * np.pi) * np.cos((col_pos / tile_size) * 2 * np.pi)

    return pe


tile_size = 16
max_position = tile_size * tile_size
d_model = 128

pe = sinusoidal_positional_encoding(max_position, d_model, tile_size)

plt.figure(figsize=(12, 8))
plt.imshow(pe, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
plt.colorbar()
plt.title("Sinusoidal Positional Encoding")
plt.xlabel("Dimension")
plt.ylabel("Position")
plt.tight_layout()
plt.savefig("positional_encoding.png")
