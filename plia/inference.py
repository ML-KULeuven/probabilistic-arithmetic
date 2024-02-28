import math
import numpy as np
import tensorflow as tf

from .pint import PInt, PIverson
from .arithmetics import EPSILON


def log_expectation(x):
    if isinstance(x, bool) and x == False:
        return -np.inf
    elif isinstance(x, PInt):
        values = tf.math.log(tf.range(x.lower, x.upper + 1) + EPSILON)
        E = values + x.logprobs
        return tf.reduce_logsumexp(E, axis=-1)
    elif isinstance(x, PIverson):
        E = tf.reduce_logsumexp(x.logprobs, axis=-1)
        if x.negated:
            E = log1mexp(E)
        return E


def log1mexp(x):
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    https://github.com/pytorch/pytorch/issues/39242
    """
    mask = -math.log(2) < x  # x < 0
    return tf.where(
        mask,
        (
            tf.math.log(tf.math.expm1(-x)),
            tf.math.log1p(tf.math.exp(-x)),
        ),
    )


def ifthenelse(variable, lt, tbranch, fbranch):
    if variable.lower < lt and variable.upper >= lt:
        logprob_true = log_expectation(variable < lt)
        logprob_false = log1mexp(logprob_true)

        tvar = PInt(variable[..., : lt - variable.lower], variable.lower)
        fvar = PInt(variable[..., lt - variable.lower :], lt)

        tvar = logprob_true + tbranch(tvar).logits
        fvar = logprob_false + fbranch(fvar).logits

        # TODO align domains using padding
        exit()

        logits = tf.math.logaddexp(tvar, fvar)

        # TODO figure out lower
        exit()

        return PInt(logits, lower)

    else:
        raise NotImplementedError()
