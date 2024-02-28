import numpy as np
import tensorflow as tf

from typing import Union, List
from plia.arithmetics import addPIntPInt, mulitplyPIntInt, ltz, eqz


class PArray:
    def __init__(self, logits, lower):
        self.logits = logits
        self.lower = lower

    @property
    def cardinality(self):
        return self.logits.shape[-1]

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
            logits, lower = addPIntPInt(self, other)
            return PInt(logits, lower=lower, normalize=False)
        elif isinstance(other, int):
            return PInt(self.logits, lower=self.lower + other)
        else:
            raise NotImplementedError()

    def __neg__(self):
        return PInt(self.logits[::-1], lower=-self.upper)

    def __sub__(self, other):
        if isinstance(other, (PInt, int)):
            return self + (-other)
        else:
            raise NotImplementedError()

    def __mul__(self, other: int):
        if isinstance(other, int):
            logits, lower = mulitplyPIntInt(self, other)
            return PInt(logits, lower, normalize=False)
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
            x = self - other
            if x.lower >= 0:
                return False
            else:
                logits = x.logits[..., : abs(x.lower)]
                return PIverson(logits, x.lower)
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
            x = self - other
            if x.lower > 0 or x.upper < 0:
                return False
            else:
                logits = x.logits[..., abs(x.lower) : abs(x.lower) + 1]
            return PIverson(logits, 0)
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
        return PIverson(pint.logits, print.lower, negated=True)
