# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""SAmplingHuman Assignment - VAE training loop"""

import argparse
import logging

import cv2 as cv
import torch

from usecases.assignments.samplinghuman.models.vae import AutoEncoder
from usecases.assignments.samplinghuman.utils.dataset_gen import IMAGE_CHANNELS, IMAGE_SIZE, DatasetGen
from usecases.assignments.samplinghuman.utils.io import CHEKPOINTS_DIR_NAME, VISU_DIR_NAME, create_model_run_output_dir
from usecases.assignments.samplinghuman.utils.loss import fft_loss
from usecases.assignments.samplinghuman.utils.net_utils import initialize_weights
from usecases.assignments.samplinghuman.utils.visu import visualize_image_encoder_decoder

TRAINING_STEPS: int = 20000
BATCH_SIZE: int = 128
MIXED_IMAGES_NUM: int = 1
EMBEDDING_DIM: int = 128  # multiplier of 4
TILE_SIZE: int = 16
TRAINING_LOGGING_FREQUENCY: int = 100
VALIDATION_FREQUENCY: int = 1000
OUTPUT_DIR: str = "/workspaces/metron_ai_deepforge/output"
CHECKPOINT_FREQUENCY: int = 10000
COSINE_SCHEDULER_PERIOD: int = 300
WARMUP_ITERATIONS: int = 10000

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoint", required=False, type=str, help="Model checkpoint path")
args = parser.parse_args()

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

encoder_train_dataset = DatasetGen(
    "/mnt/samplinghuman_data", batch_size=BATCH_SIZE, mixed_images_num=MIXED_IMAGES_NUM, split="train", tile_size=TILE_SIZE, use_augmentations=False
)
encoder_val_dataset = DatasetGen(
    "/mnt/samplinghuman_data", batch_size=BATCH_SIZE, mixed_images_num=MIXED_IMAGES_NUM, split="test", tile_size=TILE_SIZE, use_augmentations=False
)
val_images_num = encoder_val_dataset.get_data_images_num()
train_images_num = encoder_train_dataset.get_data_images_num()
val_iterations_num = val_images_num // BATCH_SIZE
train_iterations_num = train_images_num // BATCH_SIZE
logger.info(f"Train dataset size: {train_images_num} images / {train_iterations_num} iterations")
logger.info(f"Validation dataset size: {val_images_num} images / {val_iterations_num} iterations")
image_encoder_decoder_model = AutoEncoder(embedding_dim=EMBEDDING_DIM, device="cuda", checkpoint_path=args.checkpoint)
optimizer = torch.optim.Adam(image_encoder_decoder_model.parameters(), lr=1e-4)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_ITERATIONS)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=COSINE_SCHEDULER_PERIOD, T_mult=1, eta_min=1e-5)
bce_loss = torch.nn.BCEWithLogitsLoss()
output_dir = create_model_run_output_dir(OUTPUT_DIR)
training_log = open(f"{output_dir}/training_log.txt", "w")
val_log = open(f"{output_dir}/val_log.txt", "w")
visu_dir = f"{output_dir}/{VISU_DIR_NAME}"


for param in image_encoder_decoder_model.encoder.parameters():
    param.requires_grad = True
for param in image_encoder_decoder_model.decoder.parameters():
    param.requires_grad = False
training_mode = "encoder"

image_encoder_decoder_model.train()
if args.checkpoint is None:
    image_encoder_decoder_model.apply(initialize_weights)
for step in range(TRAINING_STEPS):
    image_encoder_decoder_model.zero_grad()
    x_train, _ = next(encoder_train_dataset)
    x_train = torch.squeeze(x_train)
    x_train = torch.transpose(x_train, 2, 1).reshape(BATCH_SIZE * (IMAGE_SIZE // TILE_SIZE) ** 2, IMAGE_CHANNELS, TILE_SIZE, TILE_SIZE).to("cuda")
    x_predicted, embedding = image_encoder_decoder_model(x_train)
    batch_bce_loss = bce_loss(x_predicted, x_train) * 10
    x_predicted = torch.nn.functional.sigmoid(x_predicted)
    fft_mse_loss = fft_loss(x_predicted, x_train)
    batch_loss = batch_bce_loss + fft_mse_loss + 0.05 * torch.norm(1.0 - embedding, p=2)
    batch_loss.backward()
    optimizer.step()
    cosine_scheduler.step() if step > WARMUP_ITERATIONS else warmup_scheduler.step()
    if step % TRAINING_LOGGING_FREQUENCY == 0:
        logger.info(
            f"Step {step}, BCE: {batch_bce_loss.item():.4f}, FFT2 MSE: {fft_mse_loss.item():.4f}, Total loss: {batch_loss.item():.4f}, latent: {embedding.mean()}"
        )
        training_log.write(f"{step},{batch_loss.item():.4f}\n")
        training_log.flush()

    if step % 300 == 0:
        if training_mode == "encoder":
            for param in image_encoder_decoder_model.encoder.parameters():
                param.requires_grad = True
            for param in image_encoder_decoder_model.decoder.parameters():
                param.requires_grad = False
            training_mode = "decoder"
        elif training_mode == "decoder":
            for param in image_encoder_decoder_model.encoder.parameters():
                param.requires_grad = False
            for param in image_encoder_decoder_model.decoder.parameters():
                param.requires_grad = True
            training_mode = "encoder"

    if step % VALIDATION_FREQUENCY == 0:
        val_batch_bce_loss = 0.0
        val_fft2_mse_loss = 0.0
        image_encoder_decoder_model.eval()
        with torch.no_grad():
            for val_step in range(val_iterations_num):
                x_val, _ = next(encoder_val_dataset)
                x_val = torch.squeeze(x_val)
                x_val = (
                    torch.transpose(x_val, 2, 1).reshape(BATCH_SIZE * (IMAGE_SIZE // TILE_SIZE) ** 2, IMAGE_CHANNELS, TILE_SIZE, TILE_SIZE).to("cuda")
                )
                x_val_predicted, val_embedding = image_encoder_decoder_model(x_val)
                val_batch_bce_loss += bce_loss(x_val_predicted, x_val).item() * 10
                x_val_predicted = torch.nn.functional.sigmoid(x_val_predicted)
                val_fft2_mse_loss += fft_loss(x_val_predicted, x_val).item()
        image_encoder_decoder_model.train()
        val_batch_bce_loss /= val_iterations_num
        val_fft2_mse_loss /= val_iterations_num
        val_loss = val_batch_bce_loss  # + val_fft2_mse_loss
        logger.info(
            f"Validation: step {step}, BCE: {val_batch_bce_loss:.4f} , FFT MSE: {val_fft2_mse_loss:.4f}, total loss: {val_loss:.4f}, latent: {val_embedding.mean()}"
        )
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
