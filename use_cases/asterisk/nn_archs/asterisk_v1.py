"""Asterisk v1
"""

import tensorflow as tf


def get_net(input):
    x = tf.keras.layers.Normalization(axis=-1, mean=0, variance=1, invert=False)(input)
    x = tf.keras.layers.Conv2D(activation="leaky_relu", filters=16, kernel_size=(7, 7), strides=(1, 1))(x)
    x = tf.keras.layers.MaxPool2D(strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(activation="leaky_relu", filters=32, kernel_size=(3, 3), strides=(1, 1))(x)
    x = tf.keras.layers.Conv2D(activation="leaky_relu", filters=64, kernel_size=(3, 3), strides=(1, 1))(x)
    x = tf.keras.layers.MaxPool2D(strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(activation="leaky_relu", filters=128, kernel_size=(3, 3), strides=(1, 1))(x)
    x = tf.keras.layers.MaxPool2D(strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(activation="leaky_relu", filters=128, kernel_size=(3, 3), strides=(1, 1))(x)
    x = tf.keras.layers.AveragePooling2D()(x)
    return x, x
