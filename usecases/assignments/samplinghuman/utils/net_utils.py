# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Network utility functions"""

from torch import nn


def initialize_weights(layer: nn.Module):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
