"""F1-Score Metric
"""

import tensorflow as tf


class F1ScoreMetric(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", dtype=None, **kwargs):
        super().__init__(name, dtype, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * precision * recall / (precision + recall)
