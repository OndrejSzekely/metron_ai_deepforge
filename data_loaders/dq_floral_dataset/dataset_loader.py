"""DQ Floral Dataset Loader
"""

from typing import Tuple, List, Any
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
        seed: int = 19031992,
    ) -> None:
        param_val.check_type(dataset_collector, DQFloralDatasetCollector)
        param_val.check_type(img_paths, List[str])
        param_val.check_type(img_labels, List[List[int]])
        param_val.check_type(shuffle, bool)
        param_val.check_type(seed, int)
        param_val.check_type(batch_size, int)
        param_val.check_parameter_value_in_range(batch_size, 0, len(img_paths))

        self.dataset_collector = dataset_collector
        self._img_paths = img_paths
        self._img_labels = img_labels
        self._records_num = len(img_paths)
        self._shuffle = shuffle
        self._batch_size = batch_size
        self.number_of_steps = int(self._records_num / self._batch_size)

    def _load_image(self, img_path_tensor: tf.Tensor, img_label_tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Dataset pipeline step which loads image.

        Args:
            img_path_tensor (tf.Tensor): img path.
            img_label_tensor (tf.Tensor): Expect dims [3].

        Returns:
            tf.data.Dataset: Ouput dims ([64, 64, 3], [3]). Image channels order is RGB.
        """
        img_binary = tf.io.read_file(img_path_tensor)
        img = tf.io.decode_png(img_binary)
        img = tf.image.resize(img, size=[64, 64])
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

        return {"input": img}, {
            "output_species": species_encoding,
            "output_open": bloom_encoding,
            "output_insect": beetle_encoding,
        }

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        dataset = tf.data.Dataset.from_tensor_slices((self._img_paths, self._img_labels))
        if self._shuffle:
            dataset = dataset.shuffle(self._records_num, reshuffle_each_iteration=True)
        dataset = dataset.map(self._load_image)
        dataset = dataset.map(self._preprocess_label)
        dataset = dataset.batch(self._batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = dataset.repeat()

        return dataset
