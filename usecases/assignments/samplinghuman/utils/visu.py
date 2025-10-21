# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Visualization module"""

import math

import numpy as np
import torch

from usecases.assignments.samplinghuman.utils.dataset_gen import IMAGE_SIZE

IMGS_PER_ROW: int = 8
ELEMENT_BORDER_SIZE: int = 10  # pixels


def visualize_batch(fragment_size: int, batch: torch.Tensor) -> np.ndarray:
    """Visualizes a batch of images in a grid format."""
    batch_border = ELEMENT_BORDER_SIZE * 3
    batch = batch.cpu().numpy()
    batch_size, channels_num, fragments_num_in_sample, _, _ = batch.shape
    fragments_in_image = (IMAGE_SIZE // fragment_size) ** 2
    images_in_sample = int(fragments_num_in_sample / fragments_in_image)
    image_rows_in_sample = math.ceil(images_in_sample / IMGS_PER_ROW)
    img_with_border_width = IMAGE_SIZE + ELEMENT_BORDER_SIZE
    img_with_border_height = IMAGE_SIZE + ELEMENT_BORDER_SIZE
    canvas_width = img_with_border_width * IMGS_PER_ROW
    one_sample_images_height = img_with_border_height * image_rows_in_sample
    fragments_per_row = canvas_width // (fragment_size + ELEMENT_BORDER_SIZE)
    fragment_rows = math.ceil(fragments_num_in_sample / fragments_per_row)
    fragment_with_border_height = fragment_size + ELEMENT_BORDER_SIZE
    fragment_with_border_width = fragment_size + ELEMENT_BORDER_SIZE
    one_sample_fragments_height = fragment_with_border_height * fragment_rows
    visualized_sample_height = one_sample_images_height + one_sample_fragments_height + batch_border
    canvas_height = visualized_sample_height * batch_size
    canvas = np.zeros((canvas_height, canvas_width, channels_num), dtype=np.uint8)

    for batch_img_idx in range(batch_size):
        canvas_batch_loc_y = visualized_sample_height * batch_img_idx
        for img_idx in range(int(images_in_sample)):
            # Visualize original images
            canvas_loc_y = canvas_batch_loc_y + img_idx // IMGS_PER_ROW * img_with_border_height
            canvas_loc_x = img_idx + (img_idx % IMGS_PER_ROW) * img_with_border_width
            for column_index in range(IMAGE_SIZE // fragment_size):
                for row_index in range(IMAGE_SIZE // fragment_size):
                    canvas[
                        canvas_loc_y + row_index * fragment_size : canvas_loc_y + (row_index + 1) * fragment_size,
                        canvas_loc_x + column_index * fragment_size : canvas_loc_x + (column_index + 1) * fragment_size,
                        :,
                    ] = (
                        np.transpose(
                            batch[batch_img_idx, :, img_idx * fragments_in_image + column_index * (IMAGE_SIZE // fragment_size) + row_index, :, :],
                            (1, 2, 0),
                        )
                        * 255
                    )
        # Visualize fragments
        canvas_loc_y = canvas_batch_loc_y + one_sample_images_height
        canvas_loc_x = 0
        for fragment_idx in range(fragments_num_in_sample):
            fragment_row_loc = (fragment_idx // fragments_per_row) * fragment_with_border_height
            fragment_col_loc = (fragment_idx % fragments_per_row) * fragment_with_border_width
            canvas[
                canvas_loc_y + fragment_row_loc : canvas_loc_y + fragment_row_loc + fragment_size,
                canvas_loc_x + fragment_col_loc : canvas_loc_x + fragment_col_loc + fragment_size,
                :,
            ] = np.transpose(batch[batch_img_idx, :, fragment_idx, :, :], (1, 2, 0)) * 255

    return canvas


def visualize_image_encoder_decoder(batch: torch.Tensor, inference_res: torch.Tensor) -> np.ndarray:
    """Visualizes original and reconstructed images side by side."""
    sample_border = ELEMENT_BORDER_SIZE * 2
    batch = batch.cpu().numpy()
    inference_res = inference_res.cpu().numpy()
    batch_size, channels_num, height, width = batch.shape
    element_width = 2 * width + ELEMENT_BORDER_SIZE + sample_border
    element_height = height + sample_border
    canvas_width = element_width * IMGS_PER_ROW
    image_rows = int(math.ceil(batch_size / IMGS_PER_ROW))
    canvas_height = element_height * image_rows
    canvas = np.zeros((canvas_height, canvas_width, channels_num), dtype=np.uint8)

    for batch_img_idx in range(batch_size):
        canvas_loc_y = (batch_img_idx // IMGS_PER_ROW) * element_height
        canvas_loc_x = (batch_img_idx % IMGS_PER_ROW) * element_width
        canvas[canvas_loc_y : canvas_loc_y + height, canvas_loc_x : canvas_loc_x + width, :] = (
            np.transpose(batch[batch_img_idx, :, :, :], (1, 2, 0))
        ) * 255
        prediction_render_position_x = canvas_loc_x + width + ELEMENT_BORDER_SIZE
        canvas[canvas_loc_y : canvas_loc_y + height, prediction_render_position_x : prediction_render_position_x + width, :] = (
            np.transpose(inference_res[batch_img_idx, :, :, :], (1, 2, 0))
        ) * 255
    return canvas
