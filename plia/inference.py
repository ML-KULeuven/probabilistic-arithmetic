import math
import numpy as np
import tensorflow as tf

from .pint import PInt, PIverson
from .arithmetics import EPSILON, logit_pad


def log_expectation(x):
    """
    Implementation of the log-expectation operator for Iversons (comparisons) of probabilistic integers.

    @param x: The probabilistic integer comparison to compute the log-expectation of

    @return: The log-expectation of the probabilistic integer comparison
    """
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
    """
    Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    https://github.com/pytorch/pytorch/issues/39242

    """
    mask = -math.log(2) < x  # x < 0
    return tf.where(
        mask,
        tf.math.log(tf.math.expm1(-x)),
        tf.math.log1p(tf.math.exp(-x)),
    )


def ifthenelse(variable, lt, tbranch, fbranch):
    """
    Implementation of the probabilistic if-then-else statement.
    Currently only linear inequality constraints are supported.

    @param variable: The probabilistic integer to branch on
    @param lt: The threshold value
    @param tbranch: The function to execute if the variable is less than the threshold
    @param fbranch: The function to execute if the variable is greater or equal to the threshold

    @return: The wegihted average of the two branches
    """
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
        upper = max(t_var.upper, f_var.upper)

        t_logits = t_var.logits + t_logprob
        f_logits = f_var.logits + f_logprob

        t_logits = logit_pad(t_logits, t_var.lower - lower, upper - t_var.upper)
        f_logits = logit_pad(f_logits, f_var.lower - lower, upper - f_var.upper)

        logits = tf.experimental.numpy.logaddexp(t_logits, f_logits)

        variable = PInt(logits, lower)

        return variable
    elif variable.lower >= lt:
        return tbranch(variable)
    elif variable.upper < lt:
        return fbranch(variable)
    else:
        raise NotImplementedError()
