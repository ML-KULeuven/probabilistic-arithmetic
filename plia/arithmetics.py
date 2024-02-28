import math
import tensorflow as tf
import numpy as np


EPSILON = tf.keras.backend.epsilon()


def log_convolution(p1, p1_padding, p2, p2_padding, signal_length):
    a1 = tf.math.reduce_max(p1, axis=-1, keepdims=True)
    a2 = tf.math.reduce_max(p2, axis=-1, keepdims=True)

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
    # return logp - tf.math.reduce_logsumexp(logp, axis=-1, keepdims=True)


def addPIntPInt(x1, x2):
    domain = x1.domain + x2.domain
    cardinality = domain.max - domain.min + 1

    padding = [[0, 0] for _ in range(len(x1.logprobs.shape) - 1)]

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

    return p, domain


# def addC2C(x1, x2):
#     domain = x1.domain + x2.domain
#     n = domain.max - domain.min + 1

#     fft1 = tf.signal.rfft(tf.cast(x1.logprobs, dtype=tf.float64), [n])
#     fft2 = tf.signal.rfft(tf.cast(x2.logprobs, dtype=tf.float64), [n])
#     fft = fft1 * fft2
#     probs = tf.signal.irfft(fft, [n])
#     probs /= tf.reduce_sum(probs)
#     probs = tf.cast(probs, dtype=tf.float32)

#     return tf.math.log(probs + EPSILON), domain


def mulitplyPIntInt(x1, x2):
    logprobs = tf.reshape(x1.logprobs, [-1, x1.logprobs.shape[-1], 1])
    output_shape = [logprobs.shape[0], x2 * logprobs.shape[1], 1]
    logprobs = tf.nn.conv1d_transpose(
        tf.cast(logprobs, dtype=tf.float32),
        tf.ones([1, 1, 1]),
        output_shape,
        strides=x2,
        padding="VALID",
    )
    logprobs = tf.where(logprobs == 0, -np.inf, logprobs)
    logprobs = logprobs[:, :, 0]
    return logprobs, x1.lower * x2


def ltz(x):
    if x.lower >= 0:
        return -np.inf
    else:
        return tf.math.reduce_logsumexp(
            x.logprobs[..., : abs(x.lower)], axis=-1, keepdims=True
        )


def eqz(x):
    if x.lower > 0 or x.upper < 0:
        return -np.inf
    else:
        return x.logprobs[..., abs(x.lower)]
