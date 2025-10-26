# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Tests for <dataset_gen> module"""

import cv2
import pytest
import torch

from usecases.assignments.samplinghuman.models.vae import VAE
from usecases.assignments.samplinghuman.utils.dataset_gen import IMAGE_CHANNELS, IMAGE_SIZE, DatasetGen
from usecases.assignments.samplinghuman.utils.visu import visualize_batch, visualize_image_encoder_decoder


@pytest.mark.visual_inspection
def test_batch_visualization():
    # GIVEN: Parameters for dataset generator and a dataset generator instance
    dataset_path = "/mnt/samplinghuman_data"
    batch_size = 1
    mixed_images_num = 10
    tile_size = 16
    split = "train"
    dg = DatasetGen(dataset_path=dataset_path, batch_size=batch_size, tile_size=tile_size, mixed_images_num=mixed_images_num, split=split)

    # WHEN: Getting a batch from the dataset generator
    x, _ = next(dg)

    # THEN: Visual inspection of the batch
    render = visualize_batch(fragment_size=tile_size, batch=x)
    cv2.imwrite("visualized_batch.jpg", cv2.cvtColor(render, cv2.COLOR_RGB2BGR))


@pytest.mark.visual_inspection
def test_image_encoder_decoder_visualization():
    # GIVEN: Parameters for dataset generator,a dataset generator instance and a image encoder-decoder model
    dataset_path = "/mnt/samplinghuman_data"
    batch_size = 8
    mixed_images_num = 1
    tile_size = 16
    split = "train"
    dg = DatasetGen(dataset_path=dataset_path, batch_size=batch_size, tile_size=tile_size, mixed_images_num=mixed_images_num, split=split)
    vae = VAE(embedding_dim=256)
    vae.eval()

    # WHEN: Getting a batch from the dataset generator and running inference
    x, _ = next(dg)
    x = torch.squeeze(x)
    x = torch.transpose(x, 2, 1).reshape(batch_size * (IMAGE_SIZE // tile_size) ** 2, IMAGE_CHANNELS, tile_size, tile_size)
    inference_res, mean, log_var = vae(x)

    # THEN: Visual inspection of encoder-decoder results
    render = visualize_image_encoder_decoder(x, inference_res.detach())
    cv2.imwrite("visualized_encoder_decoder.jpg", cv2.cvtColor(render, cv2.COLOR_RGB2BGR))
