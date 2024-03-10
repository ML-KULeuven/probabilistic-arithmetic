import math
import numpy as np
import tensorflow as tf

from .pint import PInt, PIverson
from .arithmetics import EPSILON, logit_pad


def log_expectation(x):
    if isinstance(x, bool) and x == False:
        return -np.inf
    elif isinstance(x, PInt):
        values = tf.math.log(tf.range(x.lower, x.upper + 1, dtype=tf.float32) + EPSILON)
        expectation = values + x.logits
        return tf.reduce_logsumexp(expectation, axis=-1)
    elif isinstance(x, PIverson):
        expectation = tf.reduce_logsumexp(x.logits, axis=-1)
        if x.negated:
            expectation = log1mexp(expectation)
        return expectation
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
        t_logits = variable.logits[..., : lt - variable.lower]
        f_logits = variable.logits[..., lt - variable.lower :]

        t_logprob = log_expectation(variable < lt)
        f_logprob = log1mexp(t_logprob)

        t_var = PInt(t_logits, variable.lower)
        f_var = PInt(f_logits, lt)

        t_var = tbranch(t_var)
        f_var = fbranch(f_var)

        lower = min(t_var.lower, f_var.lower)
        upper = max(t_var.lower, f_var.lower)

        t_logits = t_var.logits + t_logprob
        f_logits = f_var.logits + f_logprob

        t_logits = logit_pad(t_logits, t_var.lower - lower, upper - t_var.upper)
        f_logits = logit_pad(f_logits, f_var.lower - lower, upper - f_var.upper)

        logits = tf.math.log_add_exp(t_logits, f_logits)

        variable = PInt(logits, lower)

        return accumulate + variable
    # TODO double check inequalities
    elif variable.lower >= lt:
        return accumulate + tbranch(variable)
    elif variable.upper < lt:
        return accumulate + fbranch(variable)
    else:
        raise NotImplementedError()
