import numpy as np
import tensorflow as tf

from .pint import PInt, PIverson
from .tools import EPSILON


def log_expectation(x):
    if isinstance(x, bool) and x == False:
        return -np.inf
    elif isinstance(x, PInt):
        values = tf.log(tf.range(x.lower, x.upper + 1) + EPSILON)
        E = values + x.logprobs
        return tf.reduce_logsumexp(E, axis=-1)
    elif isinstance(x, PIverson):
        E = tf.reduce_logsumexp(x.logprobs, axis=-1)
        if x.negated:
            E = 1 - E
        return E
