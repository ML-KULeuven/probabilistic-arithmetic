import tensorflow as tf


from .tools import addC2C


class Categorical:
    def __init__(self, logits, lower=0, normalize=True):
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
        return tf.math.exp(self.logprobs)

    def __add__(self, other):
        if isinstance(other, Categorical):
            logprobs, lower = addC2C(self, other)
            return Categorical(logprobs, lower=lower, normalize=False)
        elif isinstance(other, int):
            return Categorical(self.logprobs, lower=self.lower + other, normalize=False)
        else:
            raise NotImplementedError()

    def __radd__(self, other):
        if isinstance(other, int):
            return self + other
        else:
            raise NotImplementedError()

    def __str__(self):
        return f"Cateogorical(lower: {self.lower}, upper: {self.upper})"

    def print_logprobs(self):
        print(f"logprobs: {self.logprobs.numpy()}")

    def print_probs(self):
        print(f"probs: {self.probs.numpy()}")
