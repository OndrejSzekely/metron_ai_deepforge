# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

N_CLASSES = 1000
IMAGE_SIZE = (64, 64)


def normalize_img(img_batch):
    with tf.device("cpu:0"):
        img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)
        normalized_tensor = img_tensor / 255.0
    return normalized_tensor


def init_augmentor():
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        # shear_range=10,
        # zoom_range=0.1,
        channel_shift_range=0.2,
        fill_mode="reflect",
    )


class Imagenet64(object):
    def __init__(self, data_path):
        self.data_path = Path(str(data_path))

        train_files = os.listdir(self.data_path / "train_data")
        x_train = []
        y_train = []
        for train_file in train_files:
            with open(self.data_path / "train_data" / train_file, "rb") as fo:
                data = pickle.load(fo)
                x = data["data"].reshape((data["data"].shape[0], 3, 64, 64)).transpose((0, 2, 3, 1))
                y = np.array(data["labels"]) - 1

                x_train.append(x)
                y_train.append(y)
        del x, y
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        assert x_train.shape[0] == len(y_train)

        with open(self.data_path / "dev_data/dev_data_batch_1", "rb") as fo:
            data = pickle.load(fo)
            x_test = data["data"].reshape((data["data"].shape[0], 3, 64, 64)).transpose((0, 2, 3, 1))
            y_test = np.array(data["labels"]) - 1

        self.data = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
        }

        n_classes = 1000
        assert len(np.unique(self.data["y_train"])) == n_classes and len(np.unique(self.data["y_train"])) >= len(np.unique(self.data["y_test"]))

    def datagen_cls(self, batch_size, ds="train", augmentation=False):
        epoch_i = 0
        ds_size = len(self.data[f"y_{ds}"])

        augmentor = init_augmentor()

        while True:
            np.random.seed(epoch_i)
            perm = np.random.permutation(ds_size)

            for i in range(0, ds_size, batch_size):
                selection = perm[i : i + batch_size]

                if len(selection) < batch_size:
                    continue

                x, y = self.data[f"x_{ds}"][selection], self.data[f"y_{ds}"][selection]

                x = normalize_img(x)

                if augmentation:
                    x, y = next(augmentor.flow(x, y, batch_size=batch_size))

                # x: images
                # y: labels - you can ignore, not important here
                yield x, y

            epoch_i += 1


if __name__ == "__main__":
    ds = Imagenet64(
        "path_to_data_folder",
        n_decomposed_features=None,
    )
    dg = ds.datagen_cls(1024, augmentation=True)

    for i in tqdm(range(1000)):
        next(dg)
