import tensorflow as tf
import einops as E
import numpy as np

from plia import PInt
from keras.layers import *


class SumClassifier(tf.keras.Model):

    def __init__(self, encoding):
        super(SumClassifier, self).__init__()
        self.encoding = encoding

        self.neural_model = DigitClassifier()
        if encoding == "sum":
            self.addition_model = MultiAddition()
        elif encoding == "carry":
            self.addition_model = CarryAddition()
        else:
            raise NotImplementedError("Encoding must be either 'sum' or 'carry'")

    def call(self, inputs, training=None, mask=None):
        b, n, d = inputs.shape[0:3]
        inputs = E.rearrange(inputs, "b n d ... -> (b n d) ...")
        inputs = tf.expand_dims(inputs, axis=-1)
        x = self.neural_model(inputs)
        x = E.rearrange(x, "(b n d) ... -> b n d ...", b=b, n=n, d=d)

        pints = []
        for number in range(n):
            for digit in range(d):
                pints.append(PInt(x[:, number, digit, ...], 0))
        return self.addition_model(pints)


class DigitClassifier(tf.keras.Model):

    def __init__(self):
        super(DigitClassifier, self).__init__()

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(6, 5, activation="relu"))
        self.model.add(tf.keras.layers.MaxPooling2D())
        self.model.add(tf.keras.layers.Conv2D(16, 5, activation="relu"))
        self.model.add(tf.keras.layers.MaxPooling2D())
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(120, activation="relu"))
        self.model.add(tf.keras.layers.Dense(84, activation="relu"))
        self.model.add(tf.keras.layers.Dense(10))

    def call(self, inputs, training=None, mask=None):
        x = self.model(inputs)
        return x


class MultiAddition(tf.keras.Model):

    def __init__(self):
        super(MultiAddition, self).__init__()

    def call(self, inputs, training=None, mask=None):
        digits_per_number = len(inputs) // 2

        c1 = inputs[:digits_per_number]
        c2 = inputs[digits_per_number:]

        number1 = 0
        number2 = 0
        for i in range(1, digits_per_number + 1):
            number1 = number1 + c1[-i] * 10 ** (i - 1)
            number2 = number2 + c2[-i] * 10 ** (i - 1)
        return number1 + number2


class CarryAddition(tf.keras.Model):

    def __init__(self):
        super(CarryAddition, self).__init__()

    def call(self, inputs, training=None, mask=None):
        digits_per_number = len(inputs) // 2

        c1 = inputs[:digits_per_number]
        c2 = inputs[digits_per_number:]

        carry = 0
        result = []
        for i in range(1, digits_per_number + 1):
            s = c1[-i] + c2[-i] + carry
            result.append(s % 10)
            carry = s // 10
        carry.logits = tf.pad(carry.logits, [[0, 0], [0, 8]], constant_values=-np.inf)
        result.append(carry)
        return result
