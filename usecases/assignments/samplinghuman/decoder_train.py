# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""SAmplingHuman Assignment - Decoder training loop"""

import argparse
import logging

import cv2 as cv
import torch

from usecases.assignments.samplinghuman.models.transformer_decoder import TransformerDecoder
from usecases.assignments.samplinghuman.models.vae import AutoEncoder
from usecases.assignments.samplinghuman.utils.dataset_gen import IMAGE_CHANNELS, IMAGE_SIZE, DatasetGen
from usecases.assignments.samplinghuman.utils.io import CHEKPOINTS_DIR_NAME, VISU_DIR_NAME, create_model_run_output_dir
from usecases.assignments.samplinghuman.utils.loss import grouping_loss
from usecases.assignments.samplinghuman.utils.net_utils import initialize_weights
from usecases.assignments.samplinghuman.utils.visu import visualize_grouping

TRAINING_STEPS: int = 200001
BATCH_SIZE: int = 8
MIXED_IMAGES_NUM: int = 10
EMBEDDING_DIM: int = 128  # multiplier of 4
TILE_SIZE: int = 16
TRAINING_LOGGING_FREQUENCY: int = 100
VALIDATION_FREQUENCY: int = 1000
OUTPUT_DIR: str = "/workspaces/metron_ai_deepforge/output"
CHECKPOINT_FREQUENCY: int = 10000
COSINE_SCHEDULER_PERIOD: int = 300
WARMUP_ITERATIONS: int = 2000
VAL_ITERATIONS_NUM: int = 20

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--vae-checkpoint", required=False, type=str, help="VAR model checkpoint path")
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
# val_iterations_num = val_images_num // BATCH_SIZE
train_iterations_num = train_images_num // BATCH_SIZE
logger.info(f"Train dataset size: {train_images_num} images / {train_iterations_num} iterations")
logger.info(f"Validation dataset size: {val_images_num} images / {VAL_ITERATIONS_NUM} iterations")
encoder = AutoEncoder(embedding_dim=EMBEDDING_DIM, device="cuda", checkpoint_path=args.vae_checkpoint)
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False
transformer_decoder = TransformerDecoder(EMBEDDING_DIM, MIXED_IMAGES_NUM).to("cuda")
optimizer = torch.optim.Adam(transformer_decoder.parameters(), lr=1e-4)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_ITERATIONS)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=COSINE_SCHEDULER_PERIOD, T_mult=1, eta_min=1e-5)

output_dir = create_model_run_output_dir(OUTPUT_DIR)
training_log = open(f"{output_dir}/training_log.txt", "w")
val_log = open(f"{output_dir}/val_log.txt", "w")
visu_dir = f"{output_dir}/{VISU_DIR_NAME}"

cross_entropy_loss = torch.nn.CrossEntropyLoss()

transformer_decoder.train()
transformer_decoder.apply(initialize_weights)
for step in range(TRAINING_STEPS):
    transformer_decoder.train()
    transformer_decoder.zero_grad()
    x_train, y_train = next(encoder_train_dataset)
    x_train = (
        torch.transpose(x_train, 2, 1)
        .reshape(BATCH_SIZE * (IMAGE_SIZE // TILE_SIZE) ** 2 * MIXED_IMAGES_NUM, IMAGE_CHANNELS, TILE_SIZE, TILE_SIZE)
        .to("cuda")
    )
    y_train = y_train.to("cuda")
    x_train_encoded = encoder.encode(x_train)
    x_train_encoded = x_train_encoded.reshape(BATCH_SIZE, (IMAGE_SIZE // TILE_SIZE) ** 2 * MIXED_IMAGES_NUM, -1)
    x_train_predicted = transformer_decoder(x_train_encoded)
    batch_bce_loss = (
        cross_entropy_loss(x_train_predicted.view(BATCH_SIZE * (IMAGE_SIZE // TILE_SIZE) ** 2 * MIXED_IMAGES_NUM, -1), y_train.flatten()) * 10
    )
    grouping_loss_val = grouping_loss(x_train_predicted.detach(), y_train.detach(), MIXED_IMAGES_NUM)
    batch_loss = batch_bce_loss
    batch_loss.backward()
    optimizer.step()
    cosine_scheduler.step() if step > WARMUP_ITERATIONS else warmup_scheduler.step()
    if step % TRAINING_LOGGING_FREQUENCY == 0:
        logger.info(f"Step {step}, CE: {batch_bce_loss.item():.4f}, Grouping Metric: {grouping_loss_val.item():.4f}")
        training_log.write(f"{step},{batch_loss.item():.4f}\n")
        training_log.flush()

    if step % VALIDATION_FREQUENCY == 0:
        val_bce_loss = 0.0
        val_grouping_loss = 0.0
        transformer_decoder.eval()
        for val_step in range(VAL_ITERATIONS_NUM):
            x_val, y_val = next(encoder_val_dataset)
            x_val = torch.squeeze(x_val)
            x_val = (
                torch.transpose(x_train, 2, 1)
                .reshape(BATCH_SIZE * (IMAGE_SIZE // TILE_SIZE) ** 2 * MIXED_IMAGES_NUM, IMAGE_CHANNELS, TILE_SIZE, TILE_SIZE)
                .to("cuda")
            )
            y_val = y_val.to("cuda")
            x_val_encoded = encoder.encode(x_val)
            x_val_encoded = x_val_encoded.reshape(BATCH_SIZE, (IMAGE_SIZE // TILE_SIZE) ** 2 * MIXED_IMAGES_NUM, -1)
            x_val_predicted = transformer_decoder(x_val_encoded)
            val_bce_loss += (
                cross_entropy_loss(x_val_predicted.view(BATCH_SIZE * (IMAGE_SIZE // TILE_SIZE) ** 2 * MIXED_IMAGES_NUM, -1), y_val.flatten()) * 10
            )
            val_grouping_loss += grouping_loss(x_val_predicted, y_val, MIXED_IMAGES_NUM)

        transformer_decoder.train()
        val_bce_loss /= VAL_ITERATIONS_NUM
        val_grouping_loss /= VAL_ITERATIONS_NUM
        logger.info(f"Validation: step {step}, CE: {val_bce_loss:.4f}, Grouping Metric: {val_grouping_loss:.4f}")
        val_log.write(f"{step},{val_bce_loss:.4f}\n")
        val_log.flush()
        grouping_visu = visualize_grouping(y_val, x_val_predicted.detach(), x_val.transpose(-3, -1))
        cv.imwrite(visu_dir + f"/grouping_visu_{step}_val.jpg", grouping_visu)
        grouping_visu = visualize_grouping(y_train, x_train_predicted.detach(), x_train.transpose(-3, -1))
        cv.imwrite(visu_dir + f"/grouping_visu_{step}_train.jpg", grouping_visu)

    if step % CHECKPOINT_FREQUENCY == 0:
        torch.save(
            transformer_decoder.state_dict(),
            f"{output_dir}/{CHEKPOINTS_DIR_NAME}/decoder_{step}.pth",
        )
