# This file is part of the Metron AI ArDaGen (https://github.com/OndrejSzekely/metron_ai_ardagen).
# Copyright (c) 2025 Ondrej Szekely.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, version 3. This program
# is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details. You should have received a copy of the GNU General Public
# License along with this program. If not, see <http://www.gnu.org/licenses/>.

import tensorflow as tf
from .regnet import *


def get_body(input):
    x = get_stage(input, stage_index=1, number_of_blocks=1, channels_num=24, group_width=8, bottleneck_ratio=1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = get_stage(x, stage_index=2, number_of_blocks=1, channels_num=56, group_width=8, bottleneck_ratio=1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = get_stage(x, stage_index=3, number_of_blocks=2, channels_num=152, group_width=8, bottleneck_ratio=1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = get_stage(x, stage_index=4, number_of_blocks=3, channels_num=256, group_width=8, bottleneck_ratio=1)
    return x


def get_net(input):
    x = tf.keras.layers.Normalization()(input)
    x = get_stem(x)
    x = get_body(x)
    x = tf.keras.layers.AvgPool2D()(x)
    return x, x, x, x
