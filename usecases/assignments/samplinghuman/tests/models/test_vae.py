# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Tests of `vae` module."""

import pytest
import torch

from usecases.assignments.samplinghuman.models.vae import VAE, VAEDecoder, VAEEncoder


@pytest.mark.unit
def test_vae_encoder_initialization():
    # GIVEN: <embedding_dim> parameter
    embedding_dim = 256

    # WHEN: Initializing the VAEEncoder
    encoder = VAEEncoder(embedding_dim)

    # THEN: The encoder should be an instance of VAEEncoder
    assert isinstance(encoder, VAEEncoder)


@pytest.mark.unit
def test_vae_encoder_forward_pass():
    # GIVEN: A VAEEncoder and a sample input tensor
    batch_size = 2
    embedding_dim = 256
    encoder = VAEEncoder(embedding_dim)
    sample_input = torch.randn(batch_size, 3, 16, 16)

    # WHEN: Performing a forward pass
    output = encoder(sample_input)

    # THEN: The output should have the expected shape
    assert output.shape == (batch_size, embedding_dim)


@pytest.mark.unit
def test_vae_decoder_initialization():
    # GIVEN: <embedding_dim> parameter
    embedding_dim = 256

    # WHEN: Initializing the VAEDecoder
    encoder = VAEDecoder(embedding_dim)

    # THEN: The encoder should be an instance of VAEDecoder
    assert isinstance(encoder, VAEDecoder)


@pytest.mark.unit
def test_vae_decoder_forward_pass():
    # GIVEN: A VAEDecoder and a sample input tensor
    batch_size = 2
    embedding_dim = 256
    decoder = VAEDecoder(embedding_dim)
    sample_input = torch.randn(batch_size, embedding_dim)

    # WHEN: Performing a forward pass
    output = decoder(sample_input)

    # THEN: The output should have the expected shape
    assert output.shape == (batch_size, 3, 16, 16)


@pytest.mark.unit
def test_vae_initialization():
    # GIVEN: <embedding_dim> parameter
    embedding_dim = 256

    # WHEN: Initializing the VAE
    vae = VAE(embedding_dim)

    # THEN: The VAE should be an instance of VAE
    assert isinstance(vae, VAE)


@pytest.mark.unit
def test_vae_forward_pass():
    # GIVEN: A VAE and a sample input tensor
    embedding_dim = 256
    batch_size = 2
    vae = VAE(embedding_dim)
    sample_input = torch.randn(batch_size, 3, 16, 16)

    # WHEN: Performing a forward pass
    output = vae(sample_input)

    # THEN: The output should have the expected shape
    assert output.shape == (batch_size, 3, 16, 16)
