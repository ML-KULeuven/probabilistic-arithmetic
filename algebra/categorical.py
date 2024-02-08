import tensorflow as tf


from .tools import addC2C, eqz, lez, ltz


class Categorical:
    def __init__(self, name, logits, lower, normalize, engine):
        self.name = name
        self.logprobs = tf.nn.log_softmax(logits, axis=-1) if normalize else logits
        self.lower = lower
        self.engine = engine

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
            return Categorical(
                logprobs, lower=lower, normalize=False, engine=self.engine
            )
        elif isinstance(other, int):
            return Categorical(
                self.logprobs,
                lower=self.lower + other,
                normalize=False,
                engine=self.engine,
            )
        else:
            raise NotImplementedError()

    def __radd__(self, other):
        if isinstance(other, int):
            return self + other
        else:
            raise NotImplementedError()

    def __neg__(self):
        return Categorical(
            tf.reverse(self.logprobs, axis=-1),
            lower=-self.upper,
            normalize=False,
            engine=self.engine,
        )

    def __sub__(self, other):
        if isinstance(other, (Categorical, int)):
            return self + (-other)
        else:
            raise NotImplementedError()

    def __rsub__(self, other):
        if isinstance(other, int):
            return -self + other
        else:
            raise NotImplementedError()

    def __lt__(self, other):
        return ltz(self - other)

    def __rlt__(self, other):
        return ltz(other - self)

    def __le__(self, other):
        return lez(self - other)

    def __rle__(self, other):
        return lez(other - self)

    def __gt__(self, other):
        return ltz(other - self)

    def __gt__(self, other):
        return ltz(self - other)

    def __ge__(self, other):
        return lez(other - self)

    def __rge__(self, other):
        return lez(self - other)

    def __eq__(self, other):
        return eqz(self - other)

    def __repr__(self):
        return f"Categorical(lower: {self.lower}, upper: {self.upper})"

    def print_logprobs(self):
        print(f"logprobs: {self.logprobs.numpy()}")

    def print_probs(self):
        print(f"probs: {self.probs.numpy()}")
