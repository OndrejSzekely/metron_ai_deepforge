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
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D
from tensorflow.keras.layers import Add, ReLU


def resnet_block(x, filters, reps, strides):
    x = projection_block(x, filters, strides)
    for _ in range(reps - 1):
        x = identity_block(x, filters)
    return x


def projection_block(tensor, filters, strides):
    # left stream
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=strides)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4 * filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)

    # right stream
    shortcut = Conv2D(filters=4 * filters, kernel_size=1, strides=strides)(tensor)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([shortcut, x])  # skip connection
    x = ReLU()(x)
    return x


def identity_block(tensor, filters):
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=1)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4 * filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)
    x = Add()([tensor, x])  # skip connection
    x = ReLU()(x)
    return x


def conv_batchnorm_relu(x, filters, kernel_size, strides=1):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


# def get_net(input):
#     x = tf.keras.layers.Normalization(axis=-1, mean=0, variance=1, invert=False)(input)
#     x = conv_batchnorm_relu(input, filters=32, kernel_size=7, strides=2)
#     x = MaxPool2D(pool_size=3, strides=2)(x)
#     x = resnet_block(x, filters=64, reps=2, strides=1)
#     x = resnet_block(x, filters=128, reps=2, strides=2)
#     x = resnet_block(x, filters=128, reps=1, strides=2)
#     x = GlobalAvgPool2D()(x)

#     return x, x


def get_net(input):
    x = tf.keras.layers.Normalization(axis=-1, mean=0, variance=1, invert=False)(input)
    x = conv_batchnorm_relu(input, filters=32, kernel_size=7, strides=2)
    x = MaxPool2D(pool_size=3, strides=2)(x)
    x = resnet_block(x, filters=64, reps=1, strides=2)
    x = resnet_block(x, filters=64, reps=1, strides=2)
    x = resnet_block(x, filters=128, reps=1, strides=2)
    x = resnet_block(x, filters=128, reps=1, strides=2)
    species_attach = resnet_block(x, filters=128, reps=1, strides=2)
    species_attach = resnet_block(species_attach, filters=128, reps=1, strides=2)
    species_attach = GlobalAvgPool2D()(species_attach)
    bloom_attach = resnet_block(x, filters=64, reps=1, strides=2)
    bloom_attach = GlobalAvgPool2D()(bloom_attach)
    beetle_attach = resnet_block(x, filters=64, reps=1, strides=2)
    beetle_attach = GlobalAvgPool2D()(beetle_attach)

    return species_attach, bloom_attach, beetle_attach, x
