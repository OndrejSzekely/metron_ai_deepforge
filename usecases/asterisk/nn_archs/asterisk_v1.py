# This file is part of the Metron AI ArDaGen (https://github.com/OndrejSzekely/metron_ai_ardagen).
# Copyright (c) 2025 Ondrej Szekely.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, version 3. This program
# is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details. You should have received a copy of the GNU General Public
# License along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Asterisk v1"""

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
