# This file is part of the Metron AI ArDaGen (https://github.com/OndrejSzekely/metron_ai_ardagen).
# Copyright (c) 2025 Ondrej Szekely.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, version 3. This program
# is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details. You should have received a copy of the GNU General Public
# License along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Utility functions for the `Let's Build GPT` project by Andrej Karpathy."""

from enum import StrEnum

import torch
from torch import nn
from torch.nn import functional as F


class Split(StrEnum):
    """Enum to represent different data splits."""

    TRAIN = "train"
    VALIDATION = "validation"


class Dataset:
    """Class to represent a dataset with training and validation splits."""

    def __init__(self, data: torch.Tensor, split_idx: int, block_size: int, batch_size: int):
        self.train_data = data[:split_idx]
        self.val_data = data[split_idx:]
        self.block_size = block_size
        self.batch_size = batch_size

    def get_batch(self, split: Split) -> tuple[torch.Tensor, torch.Tensor]:
        """Get validation data based on the split."""
        data = self.val_data if split == Split.VALIDATION else self.train_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x, y


class BigramLanguageModel(nn.Module):
    """A simple bigram language model."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model."""
        logits = self.token_embedding_table(idx)

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss
