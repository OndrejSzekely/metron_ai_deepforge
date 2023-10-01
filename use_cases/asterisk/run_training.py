"""Training script.
"""
import logging
from typing import Final, Tuple, Any
from os import path
from sklearn.model_selection import KFold
import tensorflow as tf
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from data_loaders.dq_floral_dataset.dataset_collector import DQFloralDatasetCollector
from data_loaders.dq_floral_dataset.dataset_loader import DQFloralDatasetLoader
import utils.logging_functions as log_funcs

DATASET_PATH: Final[str] = "/dq_floral_dataset"
NUM_OF_CROSS_VAL_SPLITS: Final[int] = 7
TRAINING_BATCH_SIZE: Final[int] = 2
SKIP_K_FOLD: Final[bool] = True
OUTPUT_FOLDER: Final[str] = "/outputs"
EPOCHS_FULL_TRAINING: Final[int] = 30
EPOCHS_KFOLD_TRAINING: Final[int] = 30


def build_network() -> Any:
    """Builds the network.

    Returns:
        Any: Built model
    """
    input_layer = Input(shape=(64, 64, 3), name="input")

    # TODO expose and change
    net_body = tf.keras.layers.Conv2D(activation="relu", filters=32, kernel_size=(8, 8), strides=(4, 4))(input_layer)
    layer_for_classifier_attachment = net_body
    layer_for_embedding_attachment = net_body

    embedding_ouput_layer = Flatten(name="output_embedding")(layer_for_embedding_attachment)
    layers_for_classification_flattened = Flatten()(layer_for_classifier_attachment)
    species_classifier = Dense(units=6, activation="sigmoid", name="output_species")(
        layers_for_classification_flattened
    )
    bloom_classifier = Dense(units=2, activation="sigmoid", name="output_open")(layers_for_classification_flattened)
    beetle_classifier = Dense(units=2, activation="sigmoid", name="output_insect")(layers_for_classification_flattened)

    model = Model(
        inputs=input_layer, outputs=[species_classifier, bloom_classifier, beetle_classifier, embedding_ouput_layer]
    )

    return model


def main():
    logging.root.setLevel(logging.INFO)

    # dataset collection and splits setup
    dataset_collector = DQFloralDatasetCollector(DATASET_PATH)
    collected_dataset = dataset_collector.get_dataset()
    kfold = KFold(n_splits=NUM_OF_CROSS_VAL_SPLITS, shuffle=True, random_state=dataset_collector.seed)

    if not SKIP_K_FOLD:
        for split_ind, (train_split_indices, val_split_indices) in enumerate(
            kfold.split(collected_dataset["img_paths"])
        ):
            logging.info("/" * 50 + f" K-fold split {split_ind + 1} of {NUM_OF_CROSS_VAL_SPLITS} " + "/" * 50)

            # get training and validation dataset
            train_data_img_paths = [collected_dataset["img_paths"][ind] for ind in train_split_indices]
            train_data_img_labels = [collected_dataset["labels"][ind] for ind in train_split_indices]
            val_data_img_paths = [collected_dataset["img_paths"][ind] for ind in val_split_indices]
            val_data_img_labels = [collected_dataset["labels"][ind] for ind in val_split_indices]

            log_funcs.log_kfold_split_info(len(train_data_img_paths), len(val_data_img_paths))

            # create TF Dataset
            train_dataset_loader = DQFloralDatasetLoader(
                dataset_collector,
                train_data_img_paths,
                train_data_img_labels,
                shuffle=True,
                batch_size=TRAINING_BATCH_SIZE,
            )
            train_data = train_dataset_loader()
            val_dataset_loader = DQFloralDatasetLoader(
                dataset_collector,
                val_data_img_paths,
                val_data_img_labels,
                shuffle=False,
                batch_size=TRAINING_BATCH_SIZE,
            )
            val_data = val_dataset_loader()

            # build the model
            model = build_network()
            logging.info(model.summary())

            # define optimizer
            optimizer = tf.keras.optimizers.SGD(lr=0.001)

            # compile model
            model.compile(
                optimizer=optimizer,
                loss={
                    "output_species": "binary_crossentropy",
                    "output_open": "binary_crossentropy",
                    "output_insect": "binary_crossentropy",
                },
            )

            # train the model
            model.fit(
                train_data,
                epochs=EPOCHS_KFOLD_TRAINING,
                steps_per_epoch=train_dataset_loader.number_of_steps,
                validation_data=val_data,
            )

    # RUN FULL TRAINING

    # create TF Dataset
    train_dataset_loader = DQFloralDatasetLoader(
        dataset_collector,
        collected_dataset["img_paths"],
        collected_dataset["labels"],
        shuffle=True,
        batch_size=TRAINING_BATCH_SIZE,
    )
    train_data = train_dataset_loader()

    # build the model
    model = build_network()
    logging.info(model.summary())

    # define optimizer
    optimizer = tf.keras.optimizers.SGD(lr=0.001)

    # compile model
    model.compile(
        optimizer=optimizer,
        loss={
            "output_species": "binary_crossentropy",
            "output_open": "binary_crossentropy",
            "output_insect": "binary_crossentropy",
        },
    )

    # train the model
    model.fit(train_data, epochs=EPOCHS_FULL_TRAINING, steps_per_epoch=train_dataset_loader.number_of_steps)
    model.save(path.join(OUTPUT_FOLDER, "asterisk.h5"))


main()
