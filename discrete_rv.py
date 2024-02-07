import numpy as np
import tensorflow as tf

from integer_interval import IntegerInterval


EPS = 1e-16

class DiscreteRV:

    def __init__(self, probs: [tf.Tensor, list[float]], domain_min: int, domain_max: int, log_space: bool = True):
        self.probs = tf.constant(probs)
        self.domain = IntegerInterval(domain_min, domain_max)
        self.log_space = log_space
        # TODO: add separate id for zero entry and propagate through operations. Avoids linear/log time search for zero.

    def __add__(self, other: [int, float, 'DiscreteRV']):
        if isinstance(other, DiscreteRV):
            """ Assuming the domain is connected, we update min and max to find the new domain in constant time """
            domain = self.domain + other.domain
            n = domain.max - domain.min + 1

            if self.log_space:
                log_p1 = tf.math.log(self.probs + EPS)
                log_p2 = tf.math.log(other.probs + EPS)

                log_fft1 = tf.signal.rfft(log_p1, [n])
                log_fft2 = tf.signal.rfft(log_p2, [n])
                log_fft = log_fft1 * log_fft2
                log_probs = tf.signal.irfft(log_fft, [n])

                log_sum_exp = tf.reduce_logsumexp(log_probs, -1)
                log_probs -= log_sum_exp
                probs = tf.exp(log_probs)
            else:
                fft1 = tf.signal.rfft(tf.cast(self.probs, dtype=tf.float64), [n])
                fft2 = tf.signal.rfft(tf.cast(other.probs, dtype=tf.float64), [n])
                fft = fft1 * fft2
                probs = tf.signal.irfft(fft, [n])
                probs /= tf.reduce_sum(probs)
                probs = tf.cast(probs, dtype=tf.float32)

            return DiscreteRV(probs, domain.min, domain.max)
        elif isinstance(other, (int, float)):
            domain = self.domain + other
            return DiscreteRV(self.probs, domain.min, domain.max)
        else:
            raise NotImplementedError("You can only add a DiscreteRV or an integer to a DiscreteRV")

    def find_zero(self, interval: IntegerInterval):
        """ While we could implement this with recursion in logN, the linear time approach is just easier. """
        for i in range(interval.max - interval.min + 1):
            if interval.min + i == 0:
                return i
        return None

    def __sub__(self, other: [int, float, 'DiscreteRV']):
        if isinstance(other, (int, float, DiscreteRV)):
            return self + (-other)
        else:
            raise NotImplementedError("You can only subtract a DiscreteRV or an integer or a float from a DiscreteRV")

    def __neg__(self):
        return DiscreteRV(self.probs[::-1], -self.domain.max, -self.domain.min)

    def __mul__(self, other: [int, float]):
        if isinstance(other, (int, float)):
            probs = tf.reshape(self.probs, [-1, self.probs.shape[-1], 1])
            output_shape = [probs.shape[0], other * probs.shape[1], 1]
            probs = tf.nn.conv1d_transpose(tf.cast(probs, dtype=tf.float32), tf.ones([1, 1, 1]), output_shape, strides=other, padding='VALID')
            probs = probs[:, :, 0]
            return DiscreteRV(probs, self.domain.min * other, self.domain.max * other)
        else:
            raise NotImplementedError("You can only multiply a DiscreteRV by an integer or a float")

    def __rmul__(self, other: [int, float]):
        if isinstance(other, (int, float)):
            return self * other
        else:
            raise NotImplementedError("You can only multiply a DiscreteRV by an integer or a float")

    def __le__(self, other: [int, float, 'DiscreteRV']):
        if isinstance(other, (int, float, DiscreteRV)):
            sub = self - other
            zero_id = self.find_zero(sub.domain)
            if zero_id is None:
                return 1.
            else:
                return tf.reduce_sum(sub.probs[..., :zero_id + 1], -1)
        else:
            raise NotImplementedError("You can only compare a DiscreteRV to another DiscreteRV")

    def __lt__(self, other):
        if isinstance(other, (int, float, DiscreteRV)):
            sub = self - other
            zero_id = self.find_zero(sub.domain)
            if zero_id is None:
                return 1.
            else:
                return tf.reduce_sum(sub.probs[..., :zero_id], -1)
        else:
            raise NotImplementedError("You can only compare a DiscreteRV to another DiscreteRV")

    def __ge__(self, other):
        if isinstance(other, (int, float, DiscreteRV)):
            return 1. - (self < other)
        else:
            raise NotImplementedError("You can only compare a DiscreteRV to another DiscreteRV")

    def __gt__(self, other):
        if isinstance(other, (int, float, DiscreteRV)):
            return 1. - (self <= other)
        else:
            raise NotImplementedError("You can only compare a DiscreteRV to another DiscreteRV")

    def __eq__(self, other):
        if isinstance(other, (int, float, DiscreteRV)):
            sub = self - other
            zero_id = self.find_zero(sub.domain)
            if zero_id is None:
                return 1.
            else:
                return sub.probs[..., zero_id]
        else:
            raise NotImplementedError("You can only compare a DiscreteRV to another DiscreteRV")

    def __ne__(self, other):
        if isinstance(other, (int, float, DiscreteRV)):
            return 1. - (self == other)
        else:
            raise NotImplementedError("You can only compare a DiscreteRV to another DiscreteRV")

    def __str__(self):
        return f"DiscreteRV({self.probs.numpy()}, [{self.domain.min}, ..., {self.domain.max}])"