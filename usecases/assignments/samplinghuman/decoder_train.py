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
BATCH_SIZE: int = 1
MIXED_IMAGES_NUM: int = 2
EMBEDDING_DIM: int = 128  # multiplier of 4
TILE_SIZE: int = 16
TRAINING_LOGGING_FREQUENCY: int = 1
VALIDATION_FREQUENCY: int = 1000
OUTPUT_DIR: str = "/workspaces/metron_ai_deepforge/output"
CHECKPOINT_FREQUENCY: int = 10000
COSINE_SCHEDULER_PERIOD: int = 300
WARMUP_ITERATIONS: int = 5000
VAL_ITERATIONS_NUM: int = 20
TRAINING_MODE_CHANGE: int = 20000

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
transformer_decoder = TransformerDecoder(EMBEDDING_DIM, MIXED_IMAGES_NUM, 16, 0.0).to("cuda")
optimizer = torch.optim.AdamW(transformer_decoder.parameters(), lr=5e-6, betas=(0.9, 0.98))
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, total_iters=WARMUP_ITERATIONS)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=COSINE_SCHEDULER_PERIOD, T_mult=1, eta_min=1e-5)
bce = torch.nn.BCEWithLogitsLoss()

output_dir = create_model_run_output_dir(OUTPUT_DIR)
training_log = open(f"{output_dir}/training_log.txt", "w")
val_log = open(f"{output_dir}/val_log.txt", "w")
visu_dir = f"{output_dir}/{VISU_DIR_NAME}"

training_mode = "centroids"

x_train, y_train = next(encoder_train_dataset)
perm = torch.randperm(y_train.size(1))
x_train = x_train[:, :, perm]
y_train = y_train[:, perm]
x_train = (
    torch.transpose(x_train, 2, 1)
    .reshape(BATCH_SIZE * (IMAGE_SIZE // TILE_SIZE) ** 2 * MIXED_IMAGES_NUM, IMAGE_CHANNELS, TILE_SIZE, TILE_SIZE)
    .to("cuda")
)
y_train = y_train.to("cuda")

transformer_decoder.train()
transformer_decoder.apply(initialize_weights)
torch.nn.utils.clip_grad_norm_(transformer_decoder.parameters(), max_norm=1.0)
for step in range(TRAINING_STEPS):
    transformer_decoder.train()
    transformer_decoder.zero_grad()
    x_train_encoded = encoder.encode(x_train)
    x_train_encoded = x_train_encoded.reshape(BATCH_SIZE, (IMAGE_SIZE // TILE_SIZE) ** 2 * MIXED_IMAGES_NUM, -1)
    x_train_predicted = transformer_decoder(x_train_encoded)
    # if training_mode == "centroids":
    # batch_loss = centroid_loss(centroids_predicted, x_train_encoded, y_train, MIXED_IMAGES_NUM, EMBEDDING_DIM)
    # elif training_mode == "clustering":
    # batch_loss = focal_loss(x_train_predicted, y_train)
    batch_loss = bce(torch.squeeze(x_train_predicted, dim=-1), y_train.to(torch.float32))
    if step % TRAINING_MODE_CHANGE == 0:
        if training_mode == "centroids":
            training_mode = "clustering"

        elif training_mode == "clustering":
            training_mode = "centroids"

    with torch.no_grad():
        grouping_loss_val = grouping_loss(x_train_predicted.transpose(1, 2), y_train, MIXED_IMAGES_NUM)
    batch_loss.backward()
    optimizer.step()
    warmup_scheduler.step() if step <= WARMUP_ITERATIONS else cosine_scheduler.step()
    if step % TRAINING_LOGGING_FREQUENCY == 0:
        logger.info(f"Step {step}, Focal Loss: {batch_loss.item():.4f}, Grouping Metric: {grouping_loss_val.item():.4f}")
        training_log.write(f"{step},{batch_loss.item():.4f}\n")
        training_log.flush()

    if step % VALIDATION_FREQUENCY == 0:
        transformer_decoder.eval()
        with torch.no_grad():
            val_focal_loss = 0.0
            val_grouping_loss = 0.0
            for val_step in range(VAL_ITERATIONS_NUM):
                x_val, y_val = next(encoder_val_dataset)
                x_val = torch.squeeze(x_val)
                x_val = (
                    torch.transpose(x_val, 2, 1)
                    .reshape(BATCH_SIZE * (IMAGE_SIZE // TILE_SIZE) ** 2 * MIXED_IMAGES_NUM, IMAGE_CHANNELS, TILE_SIZE, TILE_SIZE)
                    .to("cuda")
                )
                y_val = y_val.to("cuda")
                x_val_encoded = encoder.encode(x_val)
                x_val_encoded = x_val_encoded.reshape(BATCH_SIZE, (IMAGE_SIZE // TILE_SIZE) ** 2 * MIXED_IMAGES_NUM, -1)
                x_val_predicted = transformer_decoder(x_val_encoded)
                # val_focal_loss += focal_loss(x_val_predicted, y_val)
                # val_grouping_loss += grouping_loss(x_val_predicted.transpose(1, 2), y_val, MIXED_IMAGES_NUM)

            transformer_decoder.train()
            val_focal_loss /= VAL_ITERATIONS_NUM
            val_grouping_loss /= VAL_ITERATIONS_NUM
            logger.info(f"Validation: step {step}, Focal Loss: {val_focal_loss:.4f}, Grouping Metric: {val_grouping_loss:.4f}")
            val_log.write(f"{step},{val_focal_loss:.4f}\n")
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
