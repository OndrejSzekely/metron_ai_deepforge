"""Balanced accuracy metric
"""

import tensorflow as tf


class BalancedAccuracyMetric(tf.keras.metrics.Metric):
    def __init__(self, classes_num, name="balanced_accuracy", dtype=None, **kwargs):
        super().__init__(name, dtype, **kwargs)
        self.classes_num = classes_num
        self.tps = tf.Variable([0] * classes_num, shape=self.classes_num, dtype=tf.int32)
        self.fns = tf.Variable([0] * classes_num, shape=self.classes_num, dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        prediction_one_hot = tf.one_hot(tf.argmax(y_pred, axis=1), self.classes_num, axis=1)
        tps = tf.math.reduce_sum(prediction_one_hot * y_true, axis=0)
        fns = tf.math.reduce_sum(tf.where((prediction_one_hot - y_true) < 0, 1, 0), axis=0)
        self.tps.assign(self.tps + tf.cast(tps, dtype=tf.int32))
        self.fns.assign(self.fns + tf.cast(fns, dtype=tf.int32))

    def reset_state(self):
        self.tps.assign([0] * self.classes_num)
        self.fns.assign([0] * self.classes_num)

    def result(self):
        return tf.math.reduce_mean(self.tps / (self.tps + self.fns))
