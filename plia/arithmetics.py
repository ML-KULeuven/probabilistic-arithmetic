import tensorflow as tf
import numpy as np
import math
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
    a1 = tf.math.reduce_max(p1, axis=-1, keepdims=True)
    a2 = tf.math.reduce_max(p2, axis=-1, keepdims=True)

    # TODO if we stay in log space we can probably get rid of the casting
    p1 = tf.cast(p1 - a1, dtype=tf.float64)
    p2 = tf.cast(p2 - a2, dtype=tf.float64)

    p1 = tf.math.exp(p1)
    p2 = tf.math.exp(p2)

    p1 = pad(p1, signal_length)
    p2 = pad(p2, signal_length)

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


def log_disjoint_sum(logits1, logits2):
    logits1 = tf.cast(logits1, dtype=tf.float64)
    logits2 = tf.cast(logits2, dtype=tf.float64)

    p1 = tf.math.exp(logits1)
    p2 = tf.math.exp(logits2)

    disjoint_p = p1 + p2 - p1 * p2
    logits = tf.math.log(disjoint_p + EPSILON)
    logits = tf.cast(logits, dtype=tf.float32)
    return logits - tf.math.reduce_logsumexp(logits, axis=-1, keepdims=True)


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


def orPIntPInt(x1, x2):
    lower = min(x1.lower, x2.lower)
    upper = max(x1.upper, x2.upper)

    logits1 = logit_pad(x1.logits, x1.lower - lower, upper - x1.upper)
    logits2 = logit_pad(x2.logits, x2.lower - lower, upper - x2.upper)

    logits = log_disjoint_sum(logits1, logits2)
    return logits, lower


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
