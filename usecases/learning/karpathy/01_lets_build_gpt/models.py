# This file is part of the Metron AI ArDaGen (https://github.com/OndrejSzekely/metron_ai_ardagen).
# Copyright (c) 2025 Ondrej Szekely.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, version 3. This program
# is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details. You should have received a copy of the GNU General Public
# License along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Module containing neyronal network model architectures for language modeling."""

import torch
from torch import nn
from torch.nn import functional as F


class BigramLanguageModelBase(nn.Module):
    """A simple bigram language model."""

    def __init__(self, vocab_size: int, embedding_dim: int, device: str):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim, device=device)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, device=device)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the model."""

        tok_embd = self.token_embedding_table(idx)
        logits = self.lm_head(tok_embd)

        if targets is None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate new tokens from the model."""
        # idx is (B, T) tensor of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, _ = self(idx)
            # focus on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class BigramLanguageModel(nn.Module):
    """A simple bigram language model."""

    def __init__(self, vocab_size: int, embedding_dim: int, block_size: int, device: str):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim, device=device)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim, device=device)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, device=device)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the model."""
        B, T = idx.shape

        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_embd + pos_embd
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate new tokens from the model."""
        # idx is (B, T) tensor of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, _ = self(idx)
            # focus on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class SelfAttentionHead(nn.Module):
    """One head of self-attention"""

    def __init__(self, embedding_dim: int, head_size: int, block_size: int, device: str):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False, device=device)
        self.query = nn.Linear(embedding_dim, head_size, bias=False, device=device)
        self.value = nn.Linear(embedding_dim, head_size, bias=False, device=device)
        self.register_buffer("trill", torch.tril(torch.ones(block_size, block_size, device=device)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the self-attention head."""
        B, T, C = x.shape
        k = self.key(x)  # B, block_size, head_size
        q = self.query(x)  # B, block_size, head_size
        v = self.value(x)  # B, block_size, head_size

        # compute attention scores
        wei = q @ k.transpose(-1, -2)  # (B, block_size, head_size) @ (B, head_size, block_size) -> (B, block_size, block_size)
        wei = wei * (C**-0.5)
        wei = wei.masked_fill(self.trill == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v  # (B, block_size, block_size) @ (B, block_size, head_size) -> (B, block_size, head_size)
        return out


class BigramSelfAttentionLanguageModel(nn.Module):
    """A simple bigram language model enhanced with self-attention."""

    def __init__(self, vocab_size: int, embedding_dim: int, block_size: int, device: str):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim, device=device)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim, device=device)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, device=device)
        self.sa_head = SelfAttentionHead(embedding_dim, embedding_dim, block_size, device=device)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the model."""
        B, T = idx.shape

        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_embd + pos_embd
        x = self.sa_head(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate new tokens from the model."""
        # idx is (B, T) tensor of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
