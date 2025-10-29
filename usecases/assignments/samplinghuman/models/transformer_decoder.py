import torch
from torch import nn

from usecases.assignments.samplinghuman.utils.net_utils import sinusoidal_positional_encoding


class SelfAttentionHead(nn.Module):
    def __init__(self, input_embedding_dim: int, output_embedding_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.output_embedding_dim = output_embedding_dim
        self.key = nn.Linear(input_embedding_dim, output_embedding_dim)
        self.query = nn.Linear(input_embedding_dim, output_embedding_dim)
        self.value = nn.Linear(input_embedding_dim, output_embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key = self.key(
            x
        )  # (B, fragments_num, input_embedding_dim) x (input_embedding_dim, output_embedding_dim) -> (B, fragments_num, output_embedding_dim)
        query = self.query(
            x
        )  # (B, fragments_num, input_embedding_dim) x (input_embedding_dim, output_embedding_dim) -> (B, fragments_num, output_embedding_dim)
        value = self.value(
            x
        )  # (B, fragments_num, input_embedding_dim) x (input_embedding_dim, output_embedding_dim) -> (B, fragments_num, output_embedding_dim)
        attention = query @ key.transpose(1, 2)  # (B, fragments_num, fragments_num)
        attention = attention * self.output_embedding_dim**-0.5
        attention = nn.functional.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        values = (
            attention @ value
        )  # (B, fragments_num, fragments_num) x (B, fragments_num, output_embedding_dim) -> (B, fragments_num, output_embedding_dim)
        return values


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, heads_num: int, input_embedding_dim: int, head_output_embedding_dim: int, output_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(input_embedding_dim, head_output_embedding_dim, dropout_rate) for _ in range(heads_num)])
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(head_output_embedding_dim * heads_num, output_dim)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.linear(x)
        x = self.dropout(x)

        return x


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, internal_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(input_dim, internal_dim), nn.ReLU(), nn.Linear(internal_dim, input_dim), nn.Dropout(dropout_rate))

    def forward(self, x):
        return self.ffn(x)


class TransformerBlock(nn.Module):
    def __init__(self, heads_num: int, input_embedding_dim: int, head_output_dim: int, output_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.multihead_attention = MultiHeadSelfAttention(heads_num, input_embedding_dim, head_output_dim, output_dim, dropout_rate)
        self.ln = nn.LayerNorm(output_dim)
        self.ffn = FeedForwardNet(output_dim, output_dim * 2, dropout_rate)
        self.ln2 = nn.LayerNorm(output_dim)
        self.upsampling = nn.Sequential(nn.Linear(input_embedding_dim, output_dim), nn.ReLU())

    def forward(self, x):
        # x: (B, fragments_num, embedding_dim)
        shortcut = x
        x = self.multihead_attention(x)  # (B, fragments_num, output_dim)
        x = x + self.upsampling(shortcut)
        x = self.ln(x)
        shortcut = x
        x = self.ffn(x)  # (B, fragments_num, output_dim)
        x = x + shortcut
        x = self.ln2(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim: int, groups_num: int):
        super().__init__()
        self.pe = sinusoidal_positional_encoding(embedding_dim, 2)
        self.transformer_blocks = nn.Sequential(TransformerBlock(groups_num, embedding_dim, 128, 160))
        self.linear = nn.Linear(groups_num * 16, groups_num)

    def forward(self, x):
        x = x  # + self.pe
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.linear(x)
        return x
