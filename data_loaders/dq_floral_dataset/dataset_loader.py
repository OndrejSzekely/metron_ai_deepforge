"""DQ Floral Dataset Loader
"""

from typing import Tuple, List, Any
import math
import numpy as np
import tensorflow as tf
from metron_shared import param_validators as param_val
from .dataset_collector import DQFloralDatasetCollector


class DQFloralDatasetLoader:
    """DQ Floral Dataset Loader

    Creates TF Dataset. Loads all images, tranforms images and labels.

    Data collection type: In-memory
    RAM requirements: N/A

    Attributes:
        number_of_steps (int): Computed number of steps with respect to batch size.
        dataset_collector (DQFloralDatasetCollector): Dataset collector which provides metadata about the dataset.
        _img_paths (List[str]): Image file system paths.
        _img_labels (List[int]): Image labels.
        _records_num (int): Number of images.
        _shuffle (bool): Whether to shuffle images.
        _batch_size (int): Batch size.
    """

    def __init__(
        self,
        dataset_collector: DQFloralDatasetCollector,
        img_paths: List[str],
        img_labels: List[List[int]],
        shuffle: bool,
        batch_size: int,
        augmentations: bool,
        seed: int = 19031992,
    ) -> None:
        param_val.check_type(dataset_collector, DQFloralDatasetCollector)
        param_val.check_type(img_paths, List[str])
        param_val.check_type(img_labels, List[List[int]])
        param_val.check_type(shuffle, bool)
        param_val.check_type(seed, int)
        param_val.check_type(batch_size, int)
        param_val.check_type(augmentations, bool)
        param_val.check_parameter_value_in_range(batch_size, 0, 10e4)

        self.dataset_collector = dataset_collector
        self._img_paths = img_paths
        self._img_labels = img_labels
        self._records_num = len(img_paths)
        self._shuffle = shuffle
        self._batch_size = batch_size
        self.number_of_steps = math.ceil(self._records_num / self._batch_size)
        self._augmentations = augmentations
        self._species_weights = tf.constant(
            (np.sum(np.array(dataset_collector._species_dist)) / np.array(dataset_collector._species_dist)).tolist(),
            dtype=tf.float32,
        )
        self._species_weights = self._species_weights / tf.reduce_max(self._species_weights)
        self._bloom_weights = tf.constant(
            (np.sum(np.array(dataset_collector._bloom_dist)) / np.array(dataset_collector._bloom_dist)).tolist(),
            dtype=tf.float32,
        )
        self._bloom_weights = self._bloom_weights / tf.reduce_max(self._bloom_weights)
        self._beetle_weights = tf.constant(
            (np.sum(np.array(dataset_collector._beetle_dist)) / np.array(dataset_collector._beetle_dist)).tolist(),
            dtype=tf.float32,
        )
        self._beetle_weights = self._beetle_weights / tf.reduce_max(self._beetle_weights)
        print(self._species_weights)
        print(self._bloom_weights)
        print(self._beetle_weights)

    def _load_image(self, img_path_tensor: tf.Tensor, img_label_tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Dataset pipeline step which loads image.

        Args:
            img_path_tensor (tf.Tensor): img path.
            img_label_tensor (tf.Tensor): Expect dims [3].

        Returns:
            tf.data.Dataset: Ouput dims ([64, 64, 3], [3]). Image channels order is BGR.
        """
        img_binary = tf.io.read_file(img_path_tensor)
        img = tf.io.decode_png(img_binary)
        img = tf.image.resize(img, size=[64, 64])
        img = img[..., ::-1]
        return img, img_label_tensor

    def _preprocess_label(self, img: tf.Tensor, label: tf.Tensor) -> Any:
        """Preprocess image label into one hot encoding.

        Args:
            img (tf.Tensor): Image.
            label (tf.Tensor): Label index.
        """
        species_encoding = tf.one_hot(label[0], depth=self.dataset_collector.get_number_of_species_classes())
        bloom_encoding = tf.one_hot(label[1], depth=self.dataset_collector.get_number_of_bloom_classes())
        beetle_encoding = tf.one_hot(label[2], depth=self.dataset_collector.get_number_of_beetle_classes())

        return (
            {"input": img},
            {
                "output_species": species_encoding,
                "output_open": bloom_encoding,
                "output_insect": beetle_encoding,
            },
            {
                "output_species": tf.reduce_sum(self._species_weights * species_encoding),
                "output_open": tf.reduce_sum(self._bloom_weights * bloom_encoding),
                "output_insect": tf.reduce_sum(self._beetle_weights * beetle_encoding),
            },
        )

    def _apply_augmentations(self, img: tf.Tensor, label: tf.Tensor) -> Any:
        """Apply augmentations

        Args:
            img (tf.Tensor): _description_
            label (tf.Tensor): _description_

        Returns:
            Any: _description_
        """
        seed = tf.random.uniform(shape=[2], minval=0, maxval=10000, dtype=tf.int32)

        img = tf.image.stateless_random_flip_left_right(img, seed=seed)
        img = tf.image.stateless_random_flip_up_down(img, seed=seed)
        img = tf.image.stateless_random_crop(img, size=[56, 56, 3], seed=seed)
        img = tf.image.resize(img, size=[64, 64])
        img = tf.image.stateless_random_contrast(img, lower=0.1, upper=0.4, seed=seed)
        img = tf.image.stateless_random_brightness(img, max_delta=0.3, seed=seed)
        img = tf.image.stateless_random_saturation(img, 0, 0.5, seed=seed)

        return img, label

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        dataset = tf.data.Dataset.from_tensor_slices((self._img_paths, self._img_labels))
        if self._shuffle:
            dataset = dataset.shuffle(self._records_num, reshuffle_each_iteration=True)
        dataset = dataset.map(self._load_image)
        if self._augmentations:
            dataset = dataset.map(self._apply_augmentations)
        dataset = dataset.map(self._preprocess_label)
        dataset = dataset.batch(self._batch_size, drop_remainder=False, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = dataset.repeat()

        return dataset
