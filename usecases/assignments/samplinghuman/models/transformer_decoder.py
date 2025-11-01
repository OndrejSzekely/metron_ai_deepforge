import torch
from torch import nn

from usecases.assignments.samplinghuman.utils.net_utils import sinusoidal_positional_encoding


class SelfAttentionHead(nn.Module):
    def __init__(self, input_embedding_dim: int, output_embedding_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.output_embedding_dim = output_embedding_dim
        self.key = nn.Linear(input_embedding_dim, output_embedding_dim, bias=False)
        self.query = nn.Linear(input_embedding_dim, output_embedding_dim, bias=False)
        self.value = nn.Linear(input_embedding_dim, output_embedding_dim, bias=False)
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
        attention = torch.clamp(attention, -50.0, 50.0)
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


# class DecodingHead(nn.Module):
#     def __init__(self, input_embedding_dim: int, fragments_num: int):
#         super().__init__()
#         self.key = nn.Linear(input_embedding_dim, fragments_num)
#         self.query = nn.Linear(input_embedding_dim, fragments_num)
#         self.fragments_num = fragments_num

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         key = self.key(
#             x
#         )  # (B, fragments_num, input_embedding_dim) x (input_embedding_dim, output_embedding_dim) -> (B, fragments_num, output_embedding_dim)
#         query = self.query(
#             x
#         )  # (B, fragments_num, input_embedding_dim) x (input_embedding_dim, output_embedding_dim) -> (B, fragments_num, output_embedding_dim)
#         attention = query @ key.transpose(1, 2)  # (B, fragments_num, fragments_num)
#         attention = attention * self.fragments_num**-0.5
#         attention = nn.functional.sigmoid(attention)
#         return attention


class DecodingHead(nn.Module):
    def __init__(self, input_embedding_dim: int, groups_num: int, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(input_embedding_dim, groups_num, bias=False)
        self.query = nn.Linear(input_embedding_dim, groups_num, bias=False)
        self.value = nn.Linear(input_embedding_dim, groups_num, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.groups_num = groups_num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key = self.key(x)  # (B, fragments_num, input_embedding_dim) x (input_embedding_dim, groups_num) -> (B, fragments_num, groups_num)
        query = self.query(x)  # (B, fragments_num, input_embedding_dim) x (input_embedding_dim, groups_num) -> (B, fragments_num, groups_num)
        value = self.value(x)  # (B, fragments_num, input_embedding_dim) x (input_embedding_dim, groups_num) -> (B, fragments_num, groups_num)
        attention = query @ key.transpose(1, 2)  # (B, fragments_num, fragments_num)
        attention = attention * self.groups_num**-0.5
        attention = nn.functional.softmax(attention, dim=-1)
        attention = torch.clamp(attention, -50.0, 50.0)
        attention = nn.functional.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        values = nn.functional.softmax(
            attention @ value, dim=-1
        )  # (B, fragments_num, fragments_num) x (B, fragments_num, groups_num) -> (B, fragments_num, groups_num)
        return values


class ClusterSelfAttentionHead(nn.Module):
    def __init__(self, input_embedding_dim: int, cluster_center_slots: int):
        super().__init__()
        self.cluster_center_slots = cluster_center_slots
        self.input_embedding_dim = input_embedding_dim
        self.key = nn.Linear(input_embedding_dim, cluster_center_slots, bias=False)
        self.query = nn.Linear(input_embedding_dim, cluster_center_slots, bias=False)
        self.value = nn.Linear(input_embedding_dim, cluster_center_slots, bias=False)

    def forward(self, x: torch.Tensor, fragments: torch.Tensor) -> torch.Tensor:
        # x: (B, fragments_num, input_embedding_dim)
        # fragments: (B, fragments_num, input_embedding_dim)
        query = self.query(
            x
        )  # (B, fragments_num, input_embedding_dim) x (input_embedding_dim, cluster_center_slots) -> (B, fragments_num, cluster_center_slots)
        cluster_centers = (
            query.transpose(1, 2) @ fragments
        )  # (B, cluster_center_slots, fragments_num) x (B, fragments_num, input_embedding_dim) = > (B, cluster_center_slots, input_embedding_dim)
        return cluster_centers


class ClusteringAttention(nn.Module):
    def __init__(self, input_embedding_dim: int, cluster_number: int, fragments_number: int):
        super().__init__()
        self.cluster_queries = nn.Linear(cluster_number, cluster_number, bias=False)
        self.cluster_keys = nn.Linear(cluster_number, cluster_number, bias=False)
        self.fragments_number = fragments_number

    def forward(self, x: torch.Tensor, cluster_centers: torch.Tensor) -> torch.Tensor:
        # x: (B, fragments_num, embedding_dim)
        # cluster_centers: (B, cluster_number, embedding_dim)
        cluster_center_queries = self.cluster_queries(cluster_centers.transpose(1, 2)).transpose(
            1, 2
        )  # (B, cluster_number, cluster_number) x (cluster_number, embedding_dim) -> (B, cluster_number, embedding_dim)
        cluster_center_keys = self.cluster_keys(cluster_centers.transpose(1, 2)).transpose(
            1, 2
        )  # (B, cluster_number, cluster_number) x (cluster_number, embedding_dim) -> (B, cluster_number, embedding_dim)
        cross_query = cluster_center_queries @ x.transpose(
            1, 2
        )  # (B, cluster_number, embedding_dim) x (B, embedding_dim, fragments_num) -> (B, cluster_number, fragments_num)
        cross_query = cross_query * self.fragments_number**-0.5
        cross_query = torch.clamp(cross_query, -50.0, 50.0)
        cross_query_attention = nn.functional.log_softmax(cross_query, dim=-2)
        cross_key = cluster_center_keys @ x.transpose(
            1, 2
        )  # (B, cluster_number, embedding_dim) x (B, embedding_dim, fragments_num) -> (B, cluster_number, fragments_num)
        cross_key = cross_key * self.fragments_number**-0.5
        cross_key = torch.clamp(cross_key, -50.0, 50.0)
        cross_key_attention = nn.functional.log_softmax(cross_key, dim=-2)
        values = cross_query_attention * cross_key_attention
        values = nn.functional.log_softmax(values, dim=-2)
        return values


class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim: int, groups_num: int, fragments_num: int, dropout: float = 0.0):
        super().__init__()
        self.pe = sinusoidal_positional_encoding(embedding_dim, 2)
        self.transformer_blocks = nn.Sequential(
            TransformerBlock(groups_num * 4, embedding_dim, 64, 2 * embedding_dim, dropout),
            TransformerBlock(groups_num * 3, 2 * embedding_dim, 128, 2 * embedding_dim, dropout),
            TransformerBlock(groups_num * 2, 2 * embedding_dim, 128, 2 * embedding_dim, dropout),
            TransformerBlock(groups_num, 2 * embedding_dim, 32, 1, dropout),
        )
        # self.cluster_centers = ClusterSelfAttentionHead(embedding_dim, groups_num)
        # self.decoding_head = ClusteringAttention(embedding_dim, groups_num, fragments_num)

    def forward(self, x):
        latent = x  # + self.pe
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        # cluster_centers = self.cluster_centers(x, latent)
        # x = self.decoding_head(x, cluster_centers)
        return x  # cluster_centers
