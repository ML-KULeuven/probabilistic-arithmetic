import numpy as np
import tensorflow as tf

from typing import Union, List
from plia.arithmetics import addPIntPInt, mulitplyPIntInt, ltz, eqz


EPS = tf.keras.backend.epsilon()


class PInt:
    def __init__(self, name, logits, lower, normalize):
        self.name = name
        self.logprobs = tf.nn.log_softmax(logits, axis=-1) if normalize else logits
        self.lower = lower

    @property
    def cardinality(self):
        return self.logprobs.shape[-1]

    @property
    def upper(self):
        return self.lower + self.cardinality - 1

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
            raise NotImplementedError("You can only multiply a PInt by an integer")

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self, other: int):
        return self * other

    def __lt__(self, other):
        if isinstance(other, (int, tf.Tensor, PInt)):
            return ltz(self - other)
        else:
            raise NotImplementedError("You can only compare a PInt to another PInt")

    def __le__(self, other):
        if isinstance(other, (int, tf.Tensor, PInt)):
            return ltz(self - other + 1)
        else:
            raise NotImplementedError("You can only compare a PInt to another PInt")

    def __ge__(self, other):
        if isinstance(other, (int, tf.Tensor, PInt)):
            return 1.0 - (self < other)
        else:
            raise NotImplementedError("You can only compare a PInt to another PInt")

    def __gt__(self, other):
        if isinstance(other, (int, tf.Tensor, PInt)):
            return 1.0 - (self <= other)
        else:
            raise NotImplementedError("You can only compare a PInt to another PInt")

    def __eq__(self, other):
        if isinstance(other, (int, tf.Tensor, PInt)):
            return eqz(self - other)
        else:
            raise NotImplementedError("You can only compare a PInt to another PInt")

    def __ne__(self, other):
        if isinstance(other, (int, tf.Tensor, PInt)):
            return 1.0 - (self == other)
        else:
            raise NotImplementedError("You can only compare a PInt to another PInt")

    # @property
    # def E(self):
    #     domain = tf.cast(
    #         tf.range(self.domain.min, self.domain.max + 1), dtype=tf.float64
    #     )
    #     return tf.reduce_sum(domain * tf.cast(self.probs, dtype=tf.float64), -1)

    def __str__(self):
        return f"PInt({self.probs}, [{self.domain.min}, ..., {self.domain.max}])"
