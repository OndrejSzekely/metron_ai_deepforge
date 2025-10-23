# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""SAmplingHuman Assignment - VAE training loop"""

import logging

import cv2 as cv
import torch

from usecases.assignments.samplinghuman.models.vae import VAE
from usecases.assignments.samplinghuman.utils.dataset_gen import IMAGE_CHANNELS, IMAGE_SIZE, DatasetGen
from usecases.assignments.samplinghuman.utils.io import CHEKPOINTS_DIR_NAME, VISU_DIR_NAME, create_model_run_output_dir
from usecases.assignments.samplinghuman.utils.net_utils import initialize_weights
from usecases.assignments.samplinghuman.utils.visu import visualize_image_encoder_decoder

TRAINING_STEPS: int = 30000
BATCH_SIZE: int = 256
MIXED_IMAGES_NUM: int = 10
EMBEDDING_DIM: int = 256
TILE_SIZE: int = 16
TRAINING_LOGGING_FREQUENCY: int = 100
VALIDATION_FREQUENCY: int = 1000
OUTPUT_DIR: str = "/workspaces/metron_ai_deepforge/output"
CHECKPOINT_FREQUENCY: int = 1000
WARMUP_ITERATIONS: int = 5000
COSINE_SCHEDULER_PERIOD: int = 2000

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

encoder_train_dataset = DatasetGen(
    "/mnt/samplinghuman_data", batch_size=BATCH_SIZE, mixed_images_num=1, split="train", tile_size=TILE_SIZE, use_augmentations=False
)
encoder_val_dataset = DatasetGen(
    "/mnt/samplinghuman_data", batch_size=BATCH_SIZE, mixed_images_num=1, split="test", tile_size=TILE_SIZE, use_augmentations=False
)
val_images_num = encoder_val_dataset.get_data_images_num()
train_images_num = encoder_train_dataset.get_data_images_num()
val_iterations_num = val_images_num // BATCH_SIZE
train_iterations_num = train_images_num // BATCH_SIZE
logger.info(f"Train dataset size: {train_images_num} images / {train_iterations_num} iterations")
logger.info(f"Validation dataset size: {val_images_num} images / {val_iterations_num} iterations")
image_encoder_decoder_model = VAE(embedding_dim=EMBEDDING_DIM).to("cuda")
optimizer = torch.optim.Adam(image_encoder_decoder_model.parameters(), lr=1e-3)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_ITERATIONS)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=COSINE_SCHEDULER_PERIOD, T_mult=1, eta_min=1e-5)

loss = torch.nn.MSELoss()

output_dir = create_model_run_output_dir(OUTPUT_DIR)
training_log = open(f"{output_dir}/training_log.txt", "w")
val_log = open(f"{output_dir}/val_log.txt", "w")
visu_dir = f"{output_dir}/{VISU_DIR_NAME}"

image_encoder_decoder_model.train()
image_encoder_decoder_model.apply(initialize_weights)
for step in range(TRAINING_STEPS):
    image_encoder_decoder_model.zero_grad()
    x_train, _ = next(encoder_train_dataset)
    x_train = torch.squeeze(x_train)
    x_train = torch.transpose(x_train, 2, 1).reshape(BATCH_SIZE * (IMAGE_SIZE // TILE_SIZE) ** 2, IMAGE_CHANNELS, TILE_SIZE, TILE_SIZE).to("cuda")
    x_predicted = image_encoder_decoder_model(x_train)
    batch_loss = loss(x_predicted, x_train)
    batch_loss.backward()
    optimizer.step()
    cosine_scheduler.step() if step > WARMUP_ITERATIONS else warmup_scheduler.step()
    if step % TRAINING_LOGGING_FREQUENCY == 0:
        logger.info(f"Step {step}/{TRAINING_STEPS}, loss: {batch_loss.item():.4f}")
        training_log.write(f"{step},{batch_loss.item():.4f}\n")
        training_log.flush()

    if step % VALIDATION_FREQUENCY == 0:
        val_loss = 0.0
        image_encoder_decoder_model.eval()
        for val_step in range(val_iterations_num):
            x_val, _ = next(encoder_val_dataset)
            x_val = torch.squeeze(x_val)
            x_val = torch.transpose(x_val, 2, 1).reshape(BATCH_SIZE * (IMAGE_SIZE // TILE_SIZE) ** 2, IMAGE_CHANNELS, TILE_SIZE, TILE_SIZE).to("cuda")
            x_val_predicted = image_encoder_decoder_model(x_val)
            val_loss += loss(x_val_predicted, x_val).item()
        image_encoder_decoder_model.train()
        val_loss /= val_iterations_num
        logger.info(f"Validation loss at step {step}: {val_loss:.4f}")
        val_log.write(f"{step},{val_loss:.4f}\n")
        val_log.flush()
        encoder_decoder_visu_val = visualize_image_encoder_decoder(x_val, x_val_predicted.detach())
        cv.imwrite(visu_dir + f"/encoder_decoder_visu_{step}_val.jpg", encoder_decoder_visu_val)
        encoder_decoder_visu_train = visualize_image_encoder_decoder(x_train, x_predicted.detach())
        cv.imwrite(visu_dir + f"/encoder_decoder_visu_{step}_train.jpg", encoder_decoder_visu_train)

    if step % CHECKPOINT_FREQUENCY == 0:
        torch.save(
            image_encoder_decoder_model.state_dict(),
            f"{output_dir}/{CHEKPOINTS_DIR_NAME}/vae_{step}.pth",
        )
