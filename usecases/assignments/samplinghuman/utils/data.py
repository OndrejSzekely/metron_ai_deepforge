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
        self.iterations_num = None
        self.dataset_size = None
        self.n_classes = 1000

    def load_data(self, path):
        with open(path, "rb") as fo:
            data = pickle.load(fo)
            x = data["data"].reshape((data["data"].shape[0], 3, 64, 64)).transpose((0, 2, 3, 1))
            y = np.array(data["labels"]) - 1
        return x, y

    def get_train_dataset_metadata(self):
        train_files = list(sorted(map(lambda path: self.data_path / "train_data" / path, os.listdir(self.data_path / "train_data"))))
        images_num = []
        for train_file in train_files:
            x, _ = self.load_data(train_file)
            images_num.append(len(x))
        return train_files, images_num

    def get_test_dataset_metadata(self):
        test_files = [self.data_path / "dev_data/dev_data_batch_1"]
        images_num = []
        for test_file in test_files:
            x, _ = self.load_data(test_file)
            images_num.append(len(x))
        return test_files, images_num

    def datagen_cls(self, batch_size, ds="train", augmentation=False):
        epoch_i = 0

        augmentor = init_augmentor()
        x_full, y_full = None, None
        if ds == "test":
            binary_paths, image_nums = self.get_test_dataset_metadata()
        elif ds == "train":
            binary_paths, image_nums = self.get_train_dataset_metadata()
        while True:
            np.random.seed(epoch_i)
            binaries_perm = np.random.permutation(len(binary_paths))[0 : min(len(binary_paths), 5)]
            binary_paths = [binary_paths[i] for i in binaries_perm]
            image_nums = [image_nums[i] for i in binaries_perm]

            x_list = []
            y_list = []
            for binary_path in binary_paths:
                x, y = self.load_data(binary_path)
                x_list.append(x)
                y_list.append(y)
            x_full = np.concatenate(x_list, axis=0)
            y_full = np.concatenate(y_list, axis=0)
            del x_list, y_list, x, y

            ds_size = sum(image_nums)
            iterations_num = ds_size // batch_size
            perm = np.random.permutation(ds_size)
            self.iterations_num = iterations_num
            self.dataset_size = ds_size
            for i in range(0, iterations_num, batch_size):
                selection = perm[i : i + batch_size]

                if len(selection) < batch_size:
                    continue

                x, y = x_full[selection], y_full[selection]

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
