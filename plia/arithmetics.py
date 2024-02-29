import math
import tensorflow as tf
import numpy as np


EPSILON = tf.keras.backend.epsilon()


def log_convolution(p1, p1_padding, p2, p2_padding, signal_length):
    a1 = tf.math.reduce_max(p1, axis=-1, keepdims=True)
    a2 = tf.math.reduce_max(p2, axis=-1, keepdims=True)

    # TODO if we stay in log space we can probably get rid of the casting
    p1 = tf.cast(p1 - a1, dtype=tf.float64)
    p2 = tf.cast(p2 - a2, dtype=tf.float64)

    p1 = tf.math.exp(p1)
    p2 = tf.math.exp(p2)

    p1 = tf.pad(p1, p1_padding, mode="CONSTANT", constant_values=0.0)
    p2 = tf.pad(p2, p2_padding, mode="CONSTANT", constant_values=0.0)

    """ 
    Currently convolutions in tensorflow are limited in size they can deal with. From blogposts it seems that a newer version of cudnn (8.0) can solve it for some gpus. On CPU everything is fine. 
    Specifically, MNIST addition for 7 digits is too big, even though we should have plenty of memory.

    Apparently, the biggest prime divisor of the signal length can not be bigger than 127, or at least if I can believe the internet.
    """

    p1 = tf.signal.rfft(p1, fft_length=[signal_length])
    p2 = tf.signal.rfft(p2, fft_length=[signal_length])

    p = p1 * p2

    p = tf.signal.irfft(p, fft_length=[signal_length])

    logp = tf.math.log(p + EPSILON)
    logp = tf.cast(logp, dtype=tf.float32)
    return logp + a1 + a2


def addPIntPInt(x1, x2):
    lower = x1.lower + x2.lower
    upper = x1.upper + x2.upper
    cardinality = upper - lower + 1

    padding = [[0, 0] for _ in range(len(x1.logits.shape) - 1)]

    x1_padding = tf.maximum(0, cardinality - x1.cardinality)
    x2_padding = tf.maximum(0, cardinality - x2.cardinality)

    x1_padding = padding + [[0, x1_padding]]
    x2_padding = padding + [[0, x2_padding]]

    p = log_convolution(
        x1.logits,
        x1_padding,
        x2.logits,
        x2_padding,
        cardinality,
    )

    return p, lower


def mulitplyPIntInt(x1, x2):
    logits = tf.reshape(x1.logits, [-1, x1.logits.shape[-1], 1])
    output_shape = [logits.shape[0], x2 * logits.shape[1], 1]
    logits = tf.nn.conv1d_transpose(
        tf.cast(logits, dtype=tf.float32),
        tf.ones([1, 1, 1]),
        output_shape,
        strides=x2,
        padding="VALID",
    )
    logits = tf.where(logits == 0, -np.inf, logits)
    logits = logits[:, : -(x2 - 1), 0]
    return logits, x1.lower * x2
