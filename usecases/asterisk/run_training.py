# This file is part of the Metron AI ArDaGen (https://github.com/OndrejSzekely/metron_ai_ardagen).
# Copyright (c) 2025 Ondrej Szekely.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, version 3. This program
# is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details. You should have received a copy of the GNU General Public
# License along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Training script."""

import logging
from typing import Final, Dict, Any
from os import path
from sklearn.model_selection import KFold
import tensorflow as tf
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from data_loaders.dq_floral_dataset.dataset_collector import DQFloralDatasetCollector
from data_loaders.dq_floral_dataset.dataset_loader import DQFloralDatasetLoader
import utils.logging_functions as log_funcs
from metrics.balanced_accuracy.balanced_accuracy import BalancedAccuracyMetric
from metrics.f1_score.f1_score import F1ScoreMetric
from nn_archs.regnet_200_smaller import get_net
from utils.miscellaneous import *

DATASET_PATH: Final[str] = "/dq_floral_dataset"
VAL_DATA_SIZE: Final[float] = 0.2
TRAINING_BATCH_SIZE: Final[int] = 363
TRAINING_BATCH_SIZE_FOLD: Final[int] = 291
SKIP_K_FOLD: Final[bool] = True
OUTPUT_FOLDER: Final[str] = "/outputs"
EPOCHS_FULL_TRAINING: Final[int] = 3000
EPOCHS_KFOLD_TRAINING: Final[int] = 3000
LR: Final[float] = 1e-4
TB_EXPERIMENT_NAME: Final[str] = "small_v2_1e4_final3"
LOSS: Final[Dict[str, str]] = {
    "output_species": "categorical_crossentropy",
    "output_open": "categorical_crossentropy",
    "output_insect": "categorical_crossentropy",
}
LOSS_WEIGHTS: Final[Dict[str, float]] = {
    "output_species": 1.0,
    "output_open": 1.0,
    "output_insect": 1.0,
}
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()


def build_network() -> Any:
    """Builds the network.

    Returns:
        Any: Built model
    """
    input_layer = Input(shape=(64, 64, 3), name="input")

    (
        layer_for_species_attachment,
        layer_for_bloom_attachment,
        layer_for_beetle_attachment,
        layer_for_embedding_attachment,
    ) = get_net(input_layer)

    embedding_ouput_layer = Flatten(name="output_embedding")(layer_for_embedding_attachment)
    layer_for_species_attachment = Flatten()(layer_for_species_attachment)
    layer_for_species_attachment = Dropout(0.4)(layer_for_species_attachment)
    species_classifier = Dense(units=6, activation="softmax", name="output_species")(layer_for_species_attachment)
    layer_for_bloom_attachment = Flatten()(layer_for_bloom_attachment)
    bloom_classifier = Dense(units=2, activation="softmax", name="output_open")(layer_for_bloom_attachment)
    layer_for_beetle_attachment = Flatten()(layer_for_beetle_attachment)
    beetle_classifier = Dense(units=2, activation="softmax", name="output_insect")(layer_for_beetle_attachment)

    model = Model(
        inputs=input_layer, outputs=[species_classifier, bloom_classifier, beetle_classifier, embedding_ouput_layer]
    )

    return model


def main():
    logging.root.setLevel(logging.INFO)

    # dataset collection and splits setup
    dataset_collector = DQFloralDatasetCollector(DATASET_PATH)
    collected_dataset = dataset_collector.get_dataset()
    splits_num = round(1.0 / VAL_DATA_SIZE)
    kfold = KFold(n_splits=splits_num, shuffle=True, random_state=dataset_collector.seed)

    # setting up tb logger and tb folders
    tb_logging_folder = path.join(OUTPUT_FOLDER, "logdir")
    exp_tb_logging_folder = path.join(tb_logging_folder, TB_EXPERIMENT_NAME)
    create_tb_logging_folders(tb_logging_folder, exp_tb_logging_folder)
    tb_writer = tf.summary.create_file_writer(exp_tb_logging_folder)

    # learning rate scheduler
    first_decay_steps = 60
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(LR, first_decay_steps)

    with tb_writer.as_default():
        if not SKIP_K_FOLD:
            for split_ind, (train_split_indices, val_split_indices) in enumerate(
                kfold.split(collected_dataset["img_paths"])
            ):
                logging.info("/" * 50 + f" K-fold split {split_ind + 1} of {splits_num} " + "/" * 50)

                # get training and validation dataset
                train_data_img_paths = [collected_dataset["img_paths"][ind] for ind in train_split_indices]
                train_data_img_labels = [collected_dataset["labels"][ind] for ind in train_split_indices]
                val_data_img_paths = [collected_dataset["img_paths"][ind] for ind in val_split_indices]
                val_data_img_labels = [collected_dataset["labels"][ind] for ind in val_split_indices]

                log_funcs.log_kfold_split_info(len(train_data_img_paths), len(val_data_img_paths))
                train_fold_labels_stat = compute_class_image_num(train_data_img_labels)
                tf_log_labels_stat("train", train_fold_labels_stat, split_ind)
                val_fold_labels_stat = compute_class_image_num(val_data_img_labels)
                tf_log_labels_stat("val", val_fold_labels_stat, split_ind)

                # create TF Dataset
                train_dataset_loader = DQFloralDatasetLoader(
                    dataset_collector,
                    train_data_img_paths,
                    train_data_img_labels,
                    shuffle=True,
                    batch_size=TRAINING_BATCH_SIZE_FOLD,
                    augmentations=False,
                )
                train_data = train_dataset_loader()
                val_dataset_loader = DQFloralDatasetLoader(
                    dataset_collector,
                    val_data_img_paths,
                    val_data_img_labels,
                    shuffle=False,
                    batch_size=73,
                    augmentations=False,
                )
                val_data = val_dataset_loader()

                # build the model
                model = build_network()

                # define optimizer
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn)

                # compile model
                METRICS: Final[Dict[str, Any]] = {
                    "output_species": BalancedAccuracyMetric(6),
                    "output_open": F1ScoreMetric(),
                    "output_insect": F1ScoreMetric(),
                }
                model.compile(
                    optimizer=optimizer,
                    loss=LOSS,
                    metrics=METRICS,
                    loss_weights=LOSS_WEIGHTS,
                )

                # callbacks
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

                # train the model
                kfold_history = model.fit(
                    train_data,
                    epochs=EPOCHS_KFOLD_TRAINING,
                    steps_per_epoch=train_dataset_loader.number_of_steps,
                    validation_data=val_data,
                    validation_steps=train_dataset_loader.number_of_steps,
                    # callbacks=[early_stopping]
                )
                tf_log_history("kfold", kfold_history.history, split_ind)

        # RUN FULL TRAINING

        # create TF Dataset
        train_dataset_loader = DQFloralDatasetLoader(
            dataset_collector,
            collected_dataset["img_paths"],
            collected_dataset["labels"],
            shuffle=True,
            batch_size=TRAINING_BATCH_SIZE,
            augmentations=True,
        )
        train_data = train_dataset_loader()

        # build the model
        model = build_network()
        logging.info(model.summary())

        # define optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn)

        # compile model
        METRICS: Final[Dict[str, Any]] = {
            "output_species": BalancedAccuracyMetric(6),
            "output_open": F1ScoreMetric(),
            "output_insect": F1ScoreMetric(),
        }
        model.compile(
            optimizer=optimizer,
            loss=LOSS,
            metrics=METRICS,
            loss_weights=LOSS_WEIGHTS,
        )

        # train the model
        full_train_history = model.fit(
            train_data, epochs=EPOCHS_FULL_TRAINING, steps_per_epoch=train_dataset_loader.number_of_steps, callbacks=[]
        )
        tf_log_full_history("full_training", full_train_history.history)

        tb_writer.flush()
    model.save(path.join(OUTPUT_FOLDER, "asterisk.h5"))


main()
