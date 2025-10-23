# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Dataset Generator from <Imagenet64> dataset generator"""

import numpy as np
import torch

from usecases.assignments.samplinghuman.utils.data import Imagenet64

IMAGE_SIZE = 64
IMAGE_CHANNELS = 3


class DatasetGen:
    def __init__(self, dataset_path: str, batch_size: int, mixed_images_num: int, split: str, tile_size: int, use_augmentations: bool = False):
        assert split in ["train", "test"]
        self.split = split
        self.batch_size = batch_size
        self.mixed_images_num = mixed_images_num
        self.og_dataset_gen = Imagenet64(dataset_path)
        self.dataset_gen = self.og_dataset_gen.datagen_cls(
            batch_size=self.batch_size * self.mixed_images_num, ds=split, augmentation=use_augmentations
        )
        self.tile_size = tile_size
        assert IMAGE_SIZE % tile_size == 0
        self.tiles_num = (IMAGE_SIZE // self.tile_size) ** 2

    def get_data_images_num(self):
        _, images_num = self.og_dataset_gen.get_train_dataset_metadata() if self.split == "train" else self.og_dataset_gen.get_test_dataset_metadata()
        return sum(images_num)

    def __iter__(self):
        return self

    def __next__(self):
        og_batch, _ = next(self.dataset_gen)  # batch_size * mixed_images_num
        og_batch = og_batch.numpy()
        og_batch = np.transpose(og_batch, (0, 3, 1, 2))  # to BCHW
        list_os_stripes = np.split(
            og_batch, IMAGE_SIZE // self.tile_size, axis=-1
        )  # list of [self.batch_size * self.mixed_images_num, C, H, tile_size] of IMAGE_SIZE // tile_size elements
        vertical_stack = np.concatenate(list_os_stripes, axis=-2)  # [B, C, H * (IMAGE_SIZE // tile_size), tile_size]
        tiles = np.resize(
            vertical_stack,
            (self.batch_size * self.mixed_images_num, IMAGE_CHANNELS, self.tiles_num, self.tile_size, self.tile_size),
        )  # column-major tiling of original image
        batch = np.resize(
            np.transpose(
                np.resize(tiles, (self.batch_size, self.mixed_images_num, IMAGE_CHANNELS, self.tiles_num, self.tile_size, self.tile_size)),
                (0, 2, 1, 3, 4, 5),
            ),
            (self.batch_size, IMAGE_CHANNELS, self.tiles_num * self.mixed_images_num, self.tile_size, self.tile_size),
        )  # final reshape to [B, C, mixed_images_num * num_tiles_per_image, tile_size, tile_size]

        labels = np.tile(
            np.concat([([i] * self.tiles_num) for i in range(self.mixed_images_num)]), [self.batch_size, 1]
        )  # shape [B, mixed_images_num * num_tiles_per_image]
        return torch.Tensor(batch), torch.Tensor(labels)
