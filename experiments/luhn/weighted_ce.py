import tensorflow as tf


class WeightedCE(tf.keras.losses.Loss):

    def __init__(self, weight=1.0):
        super(WeightedCE, self).__init__()
        self.weight = weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if y_true.shape[-1] > 1:
            y_true = tf.one_hot(
                tf.cast(y_true, tf.int32), y_pred.shape[-1], dtype=tf.float32
            )
        loss = tf.nn.weighted_cross_entropy_with_logits(
            y_true, y_pred, pos_weight=self.weight
        )
        return loss * self.weight
