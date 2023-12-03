"""Regnet Arch Design
"""

import tensorflow as tf
from keras.layers import Conv2D

def get_block(input, input_channels_num, stride, bottleneck_ratio, group_width):
    x = Conv2D(filters=input_channels_num/bottleneck_ratio, kernel_size=(1, 1), strides=(1, 1), activation="leaky_relu", padding="same")(input)
    x = Conv2D(filters=input_channels_num/bottleneck_ratio, kernel_size=(3, 3), strides=(stride, stride), activation="leaky_relu", groups=group_width, padding="same")(x)
    x = Conv2D(filters=input_channels_num, kernel_size=(1, 1), strides=(1, 1), activation="leaky_relu", padding="same")(x)
    shortcut = input
    if stride == 2:
        shortcut = Conv2D(filters=input_channels_num, kernel_size=(1, 1), strides=(stride, stride), activation="leaky_relu", padding="same")(input)
    x = x + shortcut
    return x
    

def get_stage(input, stage_index, channels_num, number_of_blocks, bottleneck_ratio, group_width):
    x = get_block(input, stride=2, input_channels_num=channels_num, bottleneck_ratio=bottleneck_ratio, group_width=group_width)
    for block_index in range(2, number_of_blocks+1):
        x = get_block(x, input_channels_num=channels_num, stride=1, bottleneck_ratio=bottleneck_ratio, group_width=group_width)
    return x

def get_stem(input):
    x = Conv2D(strides=2, kernel_size=(3,3), filters=24, activation="leaky_relu", padding="same")(input)
    return x