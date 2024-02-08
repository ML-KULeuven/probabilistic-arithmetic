import math
import tensorflow as tf

EPSILON = tf.keras.backend.epsilon()


def log_convolution(p1, p1_padding, p2, p2_padding, singal_length):
    a1 = tf.math.reduce_max(p1, axis=-1, keepdims=True)
    a2 = tf.math.reduce_max(p2, axis=-1, keepdims=True)

    p1 = tf.math.exp(p1 - a1)
    p2 = tf.math.exp(p2 - a2)

    p1 = tf.pad(p1, p1_padding, mode="CONSTANT", constant_values=0.0)
    p2 = tf.pad(p2, p2_padding, mode="CONSTANT", constant_values=0.0)

    assert p1.shape[-1] == singal_length
    assert p2.shape[-1] == singal_length

    p1 = tf.signal.rfft(p1, fft_length=[singal_length])
    p2 = tf.signal.rfft(p2, fft_length=[singal_length])

    p = p1 * p2
    p = tf.signal.irfft(p, fft_length=[singal_length])
    return tf.math.log(p + EPSILON) + tf.squeeze(a1, axis=-1) + tf.squeeze(a2, axis=-1)


def addC2C(x1, x2):
    assert tf.rank(x1.logprobs) == tf.rank(x2.logprobs)

    lower = x1.lower + x2.lower
    upper = x1.upper + x2.upper
    cardinality = upper - lower + 1

    padding = [[0, 0] for _ in range(tf.rank(x1.logprobs) - 1)]

    x1_padding = max(0, cardinality - x1.cardinality)
    x2_padding = max(0, cardinality - x2.cardinality)

    x1_padding = padding + [[0, x1_padding]]
    x2_padding = padding + [[0, x2_padding]]

    p = log_convolution(
        x1.logprobs,
        x1_padding,
        x2.logprobs,
        x2_padding,
        cardinality,
    )

    return p, x1.lower + x2.lower
