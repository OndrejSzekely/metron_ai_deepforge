import pytest
import torch

from usecases.assignments.samplinghuman.models.transformer_decoder import (
    FeedForwardNet,
    MultiHeadSelfAttention,
    SelfAttentionHead,
    TransformerBlock,
    TransformerDecoder,
)


@pytest.mark.unit
def test_self_attention_head():
    # GIVEN: input parameters for self-attention head and sample input
    input_embedding_dim = 32
    output_embedding_dim = 16
    batch_size = 8
    fragments_num = 10
    input_tensor = torch.ones((batch_size, fragments_num, input_embedding_dim))

    # WHEN: self-attention head initialized and called
    self_attention = SelfAttentionHead(input_embedding_dim, output_embedding_dim)
    output = self_attention(input_tensor)

    # THEN: dims match
    assert output.shape == (batch_size, fragments_num, output_embedding_dim)


@pytest.mark.unit
def test_multi_head_self_attention_head():
    # GIVEN: input parameters for multi-head self-attention head and sample input
    input_embedding_dim = 32
    head_output_embedding_dim = 8
    batch_size = 8
    fragments_num = 10
    heads_num = 8
    input_tensor = torch.ones((batch_size, fragments_num, input_embedding_dim))

    # WHEN: multi-head self-attention head initialized and called
    multi_head_self_attention = MultiHeadSelfAttention(heads_num, input_embedding_dim, head_output_embedding_dim, input_embedding_dim)
    output = multi_head_self_attention(input_tensor)

    # THEN: dims match
    assert output.shape == (batch_size, fragments_num, input_embedding_dim)


@pytest.mark.unit
def test_transformer_ffn():
    # GIVEN: input parameters for FFN and expected output
    internal_dim = 64
    dim = 32
    batch = 8
    fragments_num = 10
    input = torch.ones((batch, fragments_num, dim))

    # WHEN: FFN is instantieted and called
    ffn = FeedForwardNet(dim, internal_dim)
    output = ffn(input)

    # THEN: Dims match
    assert output.shape == (batch, fragments_num, dim)


@pytest.mark.unit
def test_transformer_block():
    # GIVEN: input parameters for TransformerBlock and expected output
    batch_size = 8
    fragments_num = 10
    heads_num = 4
    input_dim = 64
    output_dim = 128

    sample_input = torch.ones((batch_size, fragments_num, input_dim))

    # WHEN: TransformerBlock is created and called
    transformer_block = TransformerBlock(heads_num, input_dim, heads_num * input_dim, output_dim)
    output = transformer_block(sample_input)

    # THEN: Dimensions match
    assert output.shape == (batch_size, fragments_num, output_dim)


@pytest.mark.unit
def test_transformer_decoder():
    # GIVEN: parameters for transformer decoder and sample input
    batch = 6
    embedding_dim = 64
    fragments_num = 10
    groups_num = 2
    sample_input = torch.ones((batch, fragments_num, embedding_dim))

    # WHEN: Transformer initialized and called
    decoder = TransformerDecoder(embedding_dim, groups_num)
    output = decoder(sample_input)

    # THEN: dimensions match
    assert output.shape == (batch, fragments_num, groups_num)
