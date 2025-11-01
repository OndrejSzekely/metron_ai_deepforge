# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""SAmplingHuman Assignment - Decoder training loop"""

import argparse
import logging

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from usecases.assignments.samplinghuman.models.vae import AutoEncoder
from usecases.assignments.samplinghuman.utils.dataset_gen import IMAGE_CHANNELS, IMAGE_SIZE, DatasetGen
from usecases.assignments.samplinghuman.utils.io import VISU_DIR_NAME, create_model_run_output_dir

TRAINING_STEPS: int = 50
BATCH_SIZE: int = 1
MIXED_IMAGES_NUM: int = 1
EMBEDDING_DIM: int = 128  # multiplier of 4
TILE_SIZE: int = 16
TRAINING_LOGGING_FREQUENCY: int = 1
VALIDATION_FREQUENCY: int = 1000
OUTPUT_DIR: str = "/workspaces/metron_ai_deepforge/output"
CHECKPOINT_FREQUENCY: int = 10000
COSINE_SCHEDULER_PERIOD: int = 300
WARMUP_ITERATIONS: int = 5000
VAL_ITERATIONS_NUM: int = 20
TRAINING_MODE_CHANGE: int = 2000

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

output_dir = create_model_run_output_dir(OUTPUT_DIR)
training_log = open(f"{output_dir}/training_log.txt", "w")
val_log = open(f"{output_dir}/val_log.txt", "w")
visu_dir = f"{output_dir}/{VISU_DIR_NAME}"

writer = SummaryWriter(log_dir="/workspaces/metron_ai_deepforge/visu")

all_labels = np.empty((0,))
all_data = np.empty((0, 128))
for step in range(TRAINING_STEPS):
    x_train, y_train = next(encoder_train_dataset)
    x_train = (
        torch.transpose(x_train, 2, 1)
        .reshape(BATCH_SIZE * (IMAGE_SIZE // TILE_SIZE) ** 2 * MIXED_IMAGES_NUM, IMAGE_CHANNELS, TILE_SIZE, TILE_SIZE)
        .to("cuda")
    )
    y_train = y_train.numpy().squeeze()
    x_train_encoded = encoder.encode(x_train).detach().to("cpu").numpy().squeeze()
    all_labels = np.concat((all_labels, np.array([step] * 16)))
    all_data = np.concat((all_data, x_train_encoded))
writer.add_embedding(all_data, metadata=all_labels)


writer.close()
