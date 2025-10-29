# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Test of VAE training loop"""

import torch

from usecases.assignments.samplinghuman.models.vae import VAE
from usecases.assignments.samplinghuman.utils.dataset_gen import IMAGE_CHANNELS, IMAGE_SIZE, DatasetGen

TRAINING_STEPS: int = 300
BATCH_SIZE: int = 8
MIXED_IMAGES_NUM: int = 10
EMBEDDING_DIM: int = 256
TILE_SIZE: int = 16


def test_vae_training_loop():
    encoder_train_dataset = DatasetGen(
        "/mnt/samplinghuman_data", batch_size=BATCH_SIZE, mixed_images_num=1, split="train", tile_size=TILE_SIZE, use_augmentations=False
    )

    x_dummy, _ = next(encoder_train_dataset)
    x_dummy = torch.squeeze(x_dummy)
    x_dummy = torch.transpose(x_dummy, 2, 1).reshape(BATCH_SIZE * (IMAGE_SIZE // TILE_SIZE) ** 2, IMAGE_CHANNELS, TILE_SIZE, TILE_SIZE).to("cuda")
    image_encoder_decoder_model = VAE(embedding_dim=EMBEDDING_DIM, device="cuda")
    optimizer = torch.optim.Adam(image_encoder_decoder_model.parameters(), lr=1e-4)
    loss = torch.nn.MSELoss()

    image_encoder_decoder_model.train()
    for step in range(TRAINING_STEPS):
        image_encoder_decoder_model.zero_grad()
        x_predicted = image_encoder_decoder_model(x_dummy)
        batch_loss = loss(x_predicted, x_dummy)
        batch_loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"Step {step}/{TRAINING_STEPS}, loss: {batch_loss.item():.4f}")

    assert batch_loss.item() < 1e-3, "VAE training loop did not reduce loss sufficiently"
