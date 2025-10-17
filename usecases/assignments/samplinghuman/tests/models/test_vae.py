# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Tests of `vae` module."""

import pytest
import torch

from usecases.assignments.samplinghuman.models.vae import VAEDecoder, VAEEncoder


@pytest.mark.unit
def test_vae_encoder_initialization():
    # GIVEN: No specific setup

    # WHEN: Initializing the VAEEncoder
    encoder = VAEEncoder()

    # THEN: The encoder should be an instance of VAEEncoder
    assert isinstance(encoder, VAEEncoder)


@pytest.mark.unit
def test_vae_encoder_forward_pass():
    # GIVEN: A VAEEncoder and a sample input tensor
    BATCH_SIZE = 2
    encoder = VAEEncoder()
    sample_input = torch.randn(BATCH_SIZE, 3, 16, 16)

    # WHEN: Performing a forward pass
    output = encoder(sample_input)

    # THEN: The output should have the expected shape
    assert output.shape == (BATCH_SIZE, 256)


@pytest.mark.unit
def test_vae_decoder_initialization():
    # GIVEN: No specific setup

    # WHEN: Initializing the VAEDecoder
    encoder = VAEDecoder()

    # THEN: The encoder should be an instance of VAEDecoder
    assert isinstance(encoder, VAEDecoder)


@pytest.mark.unit
def test_vae_decoder_forward_pass():
    # GIVEN: A VAEDecoder and a sample input tensor
    BATCH_SIZE = 2
    decoder = VAEDecoder()
    sample_input = torch.randn(BATCH_SIZE, 256)

    # WHEN: Performing a forward pass
    output = decoder(sample_input)

    # THEN: The output should have the expected shape
    assert output.shape == (BATCH_SIZE, 3, 16, 16)
