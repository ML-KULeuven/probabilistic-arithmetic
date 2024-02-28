import tensorflow as tf

from plia.pint import PInt


class SumClassifier(tf.keras.Model):

    def __init__(self, N, batch_size=10):
        super(SumClassifier, self).__init__()
        self.N = N
        self.batch_size = batch_size

        self.neural_model = tf.keras.Sequential()
        self.neural_model.add(tf.keras.layers.Conv2D(6, 5, activation="relu"))
        self.neural_model.add(tf.keras.layers.MaxPooling2D())
        self.neural_model.add(tf.keras.layers.Conv2D(16, 5, activation="relu"))
        self.neural_model.add(tf.keras.layers.MaxPooling2D())
        self.neural_model.add(tf.keras.layers.Flatten())
        self.neural_model.add(tf.keras.layers.Dense(120, activation="relu"))
        self.neural_model.add(tf.keras.layers.Dense(84, activation="relu"))
        self.neural_model.add(tf.keras.layers.Dense(10))

        self.addition_model = MultiAddition(N)

    def call(self, inputs, training=None, mask=None):
        n1, n2 = inputs
        x = tf.concat([n1, n2], axis=1)
        x = tf.concat([x[:, i, ...] for i in range(2 * self.N)], axis=0)
        x = self.neural_model(x)
        x = [
            x[i * self.batch_size : (i + 1) * self.batch_size, :]
            for i in range(2 * self.N)
        ]
        c = [PInt(x[i], 0, 9, log_space=False) for i in range(2 * self.N)]
        return self.addition_model(c)


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
