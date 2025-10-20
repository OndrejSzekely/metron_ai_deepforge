# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Tests for <dataset_gen> module"""

import cv2
import pytest

from usecases.assignments.samplinghuman.utils.dataset_gen import DatasetGen
from usecases.assignments.samplinghuman.utils.visu import visualize_batch


@pytest.mark.visual_inspection
def test_batch_visualization():
    # GIVEN: Parameters for dataset generator and a dataset generator instance
    dataset_path = "/mnt/samplinghuman_data"
    batch_size = 4
    mixed_images_num = 10
    tile_size = 16
    split = "train"
    dg = DatasetGen(dataset_path=dataset_path, batch_size=batch_size, tile_size=tile_size, mixed_images_num=mixed_images_num, split=split)

    # WHEN: Getting a batch from the dataset generator
    x, y = next(dg)

    # THEN: Visual inspection of the batch
    render = visualize_batch(fragment_size=tile_size, batch=x)
    cv2.imwrite("visualized_batch.jpg", cv2.cvtColor(render, cv2.COLOR_RGB2BGR))
