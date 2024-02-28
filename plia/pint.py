import numpy as np
import tensorflow as tf

from typing import Union, List
from plia.tools import addPIntPInt, mulitplyPIntInt, ltz, eqz


class PArray:
    def __init__(self, logits, lower):
        self.logits = logits
        self.lower = lower

    @property
    def cardinality(self):
        return self.logprobs.shape[-1]

    @property
    def upper(self):
        return self.lower + self.cardinality - 1

    def __str__(self):
        return f"{self.__class__.__name__}(lower:{self.lower}, upper:{self.upper})"


class PInt(PArray):
    def __init__(self, logits, lower):
        super().__init__(logits, lower)

    @property
    def logprobs(self):
        return tf.nn.log_softmax(self.logits, axis=-1)

    @property
    def probs(self):
        return tf.exp(tf.cast(self.logprobs, tf.float64))

    def __add__(self, other):
        if isinstance(other, PInt):
            logprobs, lower = addPIntPInt(self, other)
            return PInt(logprobs, lower=lower, normalize=False)
        elif isinstance(other, int):
            return PInt(self.logprobs, lower=self.lower + other, normalize=False)
        else:
            raise NotImplementedError()

    def __neg__(self):
        return PInt(self.logprobs[::-1], lower=-self.upper, normalize=False)

    def __sub__(self, other):
        if isinstance(other, (PInt, int)):
            return self + (-other)
        else:
            raise NotImplementedError()

    def __mul__(self, other: int):
        if isinstance(other, int):
            logprobs, lower = mulitplyPIntInt(self, other)
            return PInt(logprobs, lower, normalize=False)
        else:
            raise NotImplementedError()

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self, other: int):
        return self * other

    def __lt__(self, other):
        if isinstance(other, (int, tf.Tensor, PInt)):
            if self.lower >= 0:
                return False
            else:
                logprobs, lower = ltz(self - other)
                return PIverson(logprobs, lower)
        else:
            raise NotImplementedError()

    def __rlt__(self, other):
        return -self < -other

    def __le__(self, other):
        self < other + 1

    def __rle__(self, other):
        -self < -other + 1

    def __gt__(self, other):
        -self < -other

    def __rgt__(self, other):
        self < other

    def __ge__(self, other):
        -self < -other + 1

    def __rge__(self, other):
        self < other + 1

    def __eq__(self, other):
        if isinstance(other, (int, tf.Tensor, PInt)):
            logprobs, lower = eqz(self - other)
            return PIverson(logprobs, lower)
        else:
            raise NotImplementedError()

    def __ne__(self, other):
        if isinstance(other, (int, tf.Tensor, PInt)):
            return ~(self == other)
        else:
            raise NotImplementedError()


class PIverson(PArray):

    def __init__(self, logits, lower, negated=False):
        super().__init__(logits, lower)
        self.negated = False

    def __ne__(self, pint):
        return PInt(pint.logprobs, print.lower, negated=True)
