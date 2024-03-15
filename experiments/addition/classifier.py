import tensorflow as tf
import einops as E

from plia import PInt


class SumClassifier(tf.keras.Model):

    def __init__(self):
        super(SumClassifier, self).__init__()

        self.neural_model = DigitClassifier()
        self.addition_model = MultiAddition()

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
