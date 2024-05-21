import tensorflow as tf
import numpy as np
import einops as E


EPSILON = tf.keras.backend.epsilon()


def pad(p, signal_length):
    padding = [[0, 0] for _ in range(len(p.shape) - 1)]
    p_padding = signal_length - p.shape[-1]
    p_padding = tf.maximum(0, padding + [[0, p_padding]])
    return tf.pad(p, p_padding, mode="CONSTANT", constant_values=0.0)


def logit_pad(logits, lower_padding, upper_padding):
    padding = [[0, 0] for _ in range(len(logits.shape) - 1)]
    padding = tf.maximum(0, padding + [[lower_padding, upper_padding]])
    return tf.pad(logits, padding, mode="CONSTANT", constant_values=-np.inf)


def log_convolution(p1, p2, signal_length):
    """
    Imlementation of summing the PMF of two probilistic integers using the fast log-conv-exp trick.

    @param p1: The PMF of the first probabilistic integer
    @param p2: The PMF of the second probabilistic integer
    @param signal_length: The length of the outcome space

    @return: The PMF of the sum of the two probabilistic integers
    """
    a1 = tf.math.reduce_max(p1, axis=-1, keepdims=True)
    a2 = tf.math.reduce_max(p2, axis=-1, keepdims=True)

    p1 = p1 - a1
    p2 = p2 - a2

    p1 = tf.cast(p1, dtype=tf.float64)
    p2 = tf.cast(p2, dtype=tf.float64)

    p1 = tf.math.exp(p1)
    p2 = tf.math.exp(p2)

    p1 = pad(p1, signal_length)
    p2 = pad(p2, signal_length)

    p1 = tf.signal.rfft(p1, fft_length=[signal_length])
    p2 = tf.signal.rfft(p2, fft_length=[signal_length])

    p = p1 * p2

    p = tf.signal.irfft(p, fft_length=[signal_length])

    logp = tf.math.log(p + EPSILON)
    logp = tf.cast(logp, dtype=tf.float32)
    return logp + a1 + a2


def multi_log_convolution(p, signal_length):
    """
    Implementation of summing the PMF of a Krat (tensor) of probabilistic integers using the fast log-conv-exp trick.

    @param p: The PMF of the probabilistic integers in a Krat
    @param signal_length: The length of the outcome space

    @return: The PMF of the sum of the probabilistic integers in the Krat
    """

    a = tf.math.reduce_max(p, axis=-1, keepdims=True)

    p = p - a
    p = tf.cast(p, dtype=tf.float64)
    p = tf.math.exp(p)

    p = pad(p, signal_length)
    p = tf.signal.rfft(p, fft_length=[signal_length])

    p = tf.math.reduce_prod(p, axis=-2)
    p = tf.signal.irfft(p, fft_length=[signal_length])

    p = tf.math.log(p + EPSILON)
    p = tf.cast(p, dtype=tf.float32)

    a = tf.math.reduce_sum(a, axis=-2)
    return p + a


def addPIntPInt(x1, x2):
    lower = x1.lower + x2.lower
    upper = x1.upper + x2.upper
    cardinality = upper - lower + 1
    p = log_convolution(x1.logits, x2.logits, cardinality)
    return p, lower


def multiplyPIntInt(x, c):
    logits = x.logits
    logits = E.rearrange(logits, "... card -> ... card 1")

    fillers = tf.ones_like(logits)
    fillers = E.repeat(fillers, "... card 1 -> ... card c", c=c - 1) * (-np.inf)

    logits = tf.concat([logits, fillers], axis=-1)
    logits = E.rearrange(logits, "... card c -> ... (card c)")[..., : -c + 1]

    return logits, x.lower * c


def integer_fill_logits(x, c):
    logits = x.logits
    lower = x.lower
    upper = x.upper

    lower_filler = (x.lower // c) * c
    upper_filler = ((x.upper + c) // c) * c - 1

    lower_filler = tf.ones(logits.shape[:-1] + (lower - lower_filler)) * (-np.inf)
    upper_filler = tf.ones(logits.shape[:-1] + (upper_filler - upper)) * (-np.inf)

    return tf.concat([lower_filler, logits, upper_filler], axis=-1)


def floordividePIntInt(x, c):
    logits = integer_fill_logits(x, c)
    logits = E.rearrange(logits, "... (card c) -> ... card c", c=c)
    logits = tf.reduce_logsumexp(logits, axis=-1)
    return logits, x.lower // c


def modPIntInt(x, c):
    logits = integer_fill_logits(x, c)
    logits = E.rearrange(logits, "... (card c) -> ... card c", c=c)
    logits = tf.reduce_logsumexp(logits, axis=-2)
    return logits, 0


def sumreduceKrat(krat):
    lower = krat.lower * krat.n_rvs
    upper = krat.upper * krat.n_rvs

    cardinality = upper - lower + 1
    p = multi_log_convolution(krat.logits, cardinality)
    return p, lower
