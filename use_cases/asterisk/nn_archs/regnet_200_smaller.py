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
