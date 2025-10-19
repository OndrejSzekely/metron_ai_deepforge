# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Tests for <dataset_gen> module"""

import pytest

from usecases.assignments.samplinghuman.utils.dataset_gen import IMAGE_CHANNELS, IMAGE_SIZE, DatasetGen


@pytest.mark.unit
def test_datasetgen_initiation():
    # GIVEN: Parameters for dataset generator
    dataset_path = "/mnt/samplinghuman_data"
    batch_size = 16
    mixed_images_num = 4
    tile_size = 16
    split = "test"
    use_augmentations = True

    # WHEN: Initiating the dataset generator
    dg = DatasetGen(
        dataset_path=dataset_path,
        batch_size=batch_size,
        mixed_images_num=mixed_images_num,
        tile_size=tile_size,
        split=split,
        use_augmentations=use_augmentations,
    )

    # THEN: The dataset generator should be created successfully
    assert isinstance(dg, DatasetGen)


@pytest.mark.unit
def test_datasetgen_yield():
    # GIVEN: Parameters for dataset generator and a dataset generator instance
    dataset_path = "/mnt/samplinghuman_data"
    batch_size = 8
    mixed_images_num = 2
    tile_size = 16
    split = "test"
    dg = DatasetGen(dataset_path=dataset_path, batch_size=batch_size, tile_size=tile_size, mixed_images_num=mixed_images_num, split=split)

    # WHEN: Getting a batch from the dataset generator
    x, y = next(dg)

    # THEN: The batch should be yielded successfully
    assert x.shape == (batch_size, IMAGE_CHANNELS, (IMAGE_SIZE // tile_size) ** 2 * mixed_images_num, tile_size, tile_size)
    assert y.shape == (batch_size, mixed_images_num * (IMAGE_SIZE // tile_size) ** 2)
