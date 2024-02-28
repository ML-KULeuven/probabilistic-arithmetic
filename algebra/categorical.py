import numpy as np
import tensorflow as tf

from typing import Union, List
from algebra.integer_interval import IntegerInterval
from algebra.tools import addC2C


EPS = tf.keras.backend.epsilon()

class Categorical:

    def __init__(self, probs: Union[tf.Tensor, List[float]], domain_min: int, domain_max: int, log_space: bool = True):
        probs = probs if isinstance(probs, tf.Tensor) else tf.convert_to_tensor(probs, dtype=tf.float32)
        self.domain = IntegerInterval(domain_min, domain_max)
        if log_space:
            self.logprobs = tf.math.log(probs + EPS)
        else:
            self.logprobs = probs

    @property
    def cardinality(self):
        return self.logprobs.shape[-1]

    @property
    def probs(self):
        return tf.exp(tf.cast(self.logprobs, tf.float64))
    
    def __add__(self, other: Union[int, 'Categorical']):
        if isinstance(other, Categorical):
            logprobs, domain = addC2C(self, other)
            return Categorical(logprobs, domain.min, domain.max, log_space=False)
        elif isinstance(other, int):
            domain = self.domain + other
            return Categorical(self.logprobs, domain.min, domain.max, log_space=False)
        else:
            raise  NotImplementedError("You can only add a Categorical or an integer to a Categorical")

    def find_zero(self, interval: IntegerInterval):
        """ While we could implement this with recursion in logN, the linear time approach is just easier. """
        for i in range(interval.max - interval.min + 1):
            if interval.min + i == 0:
                return i
        return None

    @property
    def E(self):
        domain = tf.cast(tf.range(self.domain.min, self.domain.max + 1), dtype=tf.float64)
        return tf.reduce_sum(domain * tf.cast(self.probs, dtype=tf.float64), -1)

    def __sub__(self, other: Union[int, 'Categorical']):
        if isinstance(other, (int, Categorical)):
            return self + (-other)
        else:
            raise NotImplementedError("You can only subtract a DiscreteRV or an integer or a float from a DiscreteRV")

    def __neg__(self):
        return Categorical(self.logprobs[::-1], -self.domain.max, -self.domain.min)

    def __mul__(self, other: int):
        if isinstance(other, (int)):
            logprobs = tf.reshape(self.logprobs, [-1, self.logprobs.shape[-1], 1])
            output_shape = [logprobs.shape[0], other * logprobs.shape[1], 1]
            logprobs = tf.nn.conv1d_transpose(tf.cast(logprobs, dtype=tf.float32), tf.ones([1, 1, 1]), output_shape, strides=other, padding='VALID')
            logprobs = tf.where(logprobs == 0, -np.inf, logprobs)
            logprobs = logprobs[:, :, 0]
            return Categorical(logprobs, self.domain.min * other, self.domain.max * other, log_space=False)
        else:
            raise NotImplementedError("You can only multiply a Categorical by an integer or a float")

    def __rmul__(self, other: int):
        if isinstance(other, int):
            return self * other
        else:
            raise NotImplementedError("You can only multiply a Categorical by an integer or a float")

    def __le__(self, other):
        if isinstance(other, (int, tf.Tensor, Categorical)):
            sub = self - other
            zero_id = self.find_zero(sub.domain)
            if zero_id is None:
                return 1.
            else:
                return tf.reduce_sum(sub.probs[..., :zero_id + 1], -1)
        else:
            raise NotImplementedError("You can only compare a Categorical to another Categorical")

    def __lt__(self, other):
        if isinstance(other, (int, tf.Tensor, Categorical)):
            sub = self - other
            zero_id = self.find_zero(sub.domain)
            if zero_id is None:
                return 1.
            else:
                return tf.reduce_sum(sub.probs[..., :zero_id], -1)
        else:
            raise NotImplementedError("You can only compare a Categorical to another Categorical")

    def __ge__(self, other):
        if isinstance(other, (int, tf.Tensor, Categorical)):
            return 1. - (self < other)
        else:
            raise NotImplementedError("You can only compare a Categorical to another Categorical")

    def __gt__(self, other):
        if isinstance(other, (int, tf.Tensor, Categorical)):
            return 1. - (self <= other)
        else:
            raise NotImplementedError("You can only compare a Categorical to another Categorical")

    def __eq__(self, other):
        if isinstance(other, (int, tf.Tensor, Categorical)):
            sub = self - other
            zero_id = self.find_zero(sub.domain)
            if zero_id is None:
                return 1.
            else:
                return sub.probs[..., zero_id]
        else:
            raise NotImplementedError("You can only compare a Categorical to another Categorical")

    def __ne__(self, other):
        if isinstance(other, (int, tf.Tensor, Categorical)):
            return 1. - (self == other)
        else:
            raise NotImplementedError("You can only compare a Categorical to another Categorical")

    def __str__(self):
        return f"Categorical({self.probs}, [{self.domain.min}, ..., {self.domain.max}])"