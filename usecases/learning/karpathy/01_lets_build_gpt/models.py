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

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

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

    def __init__(self, vocab_size: int, embedding_dim: int, block_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

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

    def __init__(self, embedding_dim: int, head_size: int, block_size: int, dropout_rate: float = 0.0):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer("trill", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the self-attention head."""
        B, T, C = x.shape
        k = self.key(x)  # B, block_size, head_size
        q = self.query(x)  # B, block_size, head_size
        v = self.value(x)  # B, block_size, head_size

        # compute attention scores
        wei = q @ k.transpose(-1, -2)  # (B, block_size, head_size) @ (B, head_size, block_size) -> (B, block_size, block_size)
        wei = wei * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.trill == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)  # apply dropout
        out = wei @ v  # (B, block_size, block_size) @ (B, block_size, head_size) -> (B, block_size, head_size)
        return out


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(self, embedding_dim: int, head_size: int, num_heads: int, block_size: int, dropout_rate: float = 0.0):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(embedding_dim, head_size, block_size, dropout_rate) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the multi-head self-attention layer."""
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class BigramSelfAttentionLanguageModel(nn.Module):
    """A simple bigram language model enhanced with self-attention."""

    def __init__(self, vocab_size: int, embedding_dim: int, block_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        self.sa_head = SelfAttentionHead(embedding_dim, embedding_dim, block_size)
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


class BigramMultiHeadSelfAttentionLanguageModel(nn.Module):
    """A simple bigram language model enhanced with self-attention."""

    def __init__(self, vocab_size: int, embedding_dim: int, block_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        self.sa_heads = MultiHeadSelfAttention(embedding_dim, embedding_dim // 4, 4, block_size)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the model."""
        B, T = idx.shape

        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_embd + pos_embd
        x = self.sa_heads(x)
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


class FeedForward(nn.Module):
    """a simple linear layer followed by a ReLU activation."""

    def __init__(self, embedding_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim), nn.ReLU(), nn.Linear(4 * embedding_dim, embedding_dim), nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BigramMultiHeadSelfAttentionWithFFNLanguageModel(nn.Module):
    """A simple bigram language model enhanced with self-attention and FFN."""

    def __init__(self, vocab_size: int, embedding_dim: int, block_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        self.sa_heads = MultiHeadSelfAttention(embedding_dim, embedding_dim // 4, 4, block_size)
        self.ffwd = FeedForward(embedding_dim)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the model."""
        B, T = idx.shape

        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_embd + pos_embd
        x = self.sa_heads(x)
        x = self.ffwd(x)
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


class Block(nn.Module):
    """Transformer block with multi-head self-attention and feed-forward network."""

    def __init__(self, embedding_dim: int, num_heads: int, block_size: int, dropout_rate: float):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.sa = MultiHeadSelfAttention(embedding_dim, head_size, num_heads, block_size, dropout_rate)
        self.ffwd = FeedForward(embedding_dim, dropout_rate)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT-like model"""

    def __init__(self, vocab_size: int, embedding_dim: int, block_size: int, blocks_num: int, dropout_rate: float):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        self.blocks = nn.Sequential(*[Block(embedding_dim, num_heads=6, block_size=block_size, dropout_rate=dropout_rate) for _ in range(blocks_num)])
        self.ln = nn.LayerNorm(embedding_dim)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the model."""
        B, T = idx.shape

        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln(x)
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
