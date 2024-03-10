import tensorflow as tf
import einops as E

from plia import PInt


class SumClassifier(tf.keras.Model):

    def __init__(self, D, batch_size=10):
        super(SumClassifier, self).__init__()
        self.D = D
        self.batch_size = batch_size

        self.neural_model = DigitClassifier(batch_size)
        self.addition_model = MultiAddition(D)

    def call(self, inputs, training=None, mask=None):
        # b, n, d = inputs.shape[0:3]
        # inputs = E.rearrange(inputs, "b n d ... -> (b n d) ...")
        # inputs = tf.expand_dims(inputs, axis=-1)
        # x = self.model(inputs)
        # x = E.rearrange(x, "(b n d) ... -> b i ...", b=b)

        n1, n2 = inputs
        x = tf.concat([n1, n2], axis=1)
        x = tf.concat([x[:, i, ...] for i in range(2 * self.D)], axis=0)
        x = self.neural_model(x)
        x = [
            x[i * self.batch_size : (i + 1) * self.batch_size, :]
            for i in range(2 * self.D)
        ]
        c = [PInt(x[i], 0) for i in range(2 * self.D)]
        return self.addition_model(c)


class DigitClassifier(tf.keras.Model):

    def __init__(self, batch_size=10):
        super(DigitClassifier, self).__init__()
        self.batch_size = batch_size

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
        x = self.model(x)
        return x


class MultiAddition(tf.keras.Model):

    def __init__(self, N):
        super(MultiAddition, self).__init__()
        self.N = N

    def call(self, inputs, training=None, mask=None):
        c1, c2 = inputs[: self.N], inputs[self.N :]
        number1, number2 = c1[0], c2[0]
        for i in range(1, self.N):
            number1 = number1 * 10
            number2 = number2 * 10
            number1 = number1 + c1[i]
            number2 = number2 + c2[i]
        return number1 + number2
