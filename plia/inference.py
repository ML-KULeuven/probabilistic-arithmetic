import math
import numpy as np
import tensorflow as tf

from .pint import PInt, PIverson, construct_pint
from .arithmetics import EPSILON


def log_expectation(x):
    if isinstance(x, bool) and x == False:
        return -np.inf
    elif isinstance(x, PInt):
        values = tf.math.log(tf.range(x.lower, x.upper + 1, dtype=tf.float32) + EPSILON)
        E = values + x.logits
        return tf.reduce_logsumexp(E, axis=-1)
    elif isinstance(x, PIverson):
        E = tf.reduce_logsumexp(x.logits, axis=-1)
        if x.negated:
            E = log1mexp(E)
        return E
    else:
        raise NotImplementedError()


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


def ifthenelse(variable, lt, tbranch, fbranch, accumulate):
    if variable.lower < lt and variable.upper >= lt:
        tvar = PInt(variable.logits[..., : lt - variable.lower], variable.lower)
        fvar = PInt(variable.logits[..., lt - variable.lower :], lt)

        tvar = tbranch(tvar)
        fvar = fbranch(fvar)

        return accumulate + (tvar | fvar)
    # TODO double check inequalities
    elif variable.lower >= lt:
        return accumulate + tbranch(variable)
    elif variable.upper < lt:
        return accumulate + fbranch(variable)
    else:
        raise NotImplementedError()
