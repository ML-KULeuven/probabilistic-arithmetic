import tensorflow as tf

from plia import construct_pint, ifthenelse, log_expectation
from experiments.addition.sum_classifier import DigitClassifier


class LuhnClassifier(tf.keras.Model):

    def __init__(self, length, batch_size=10):
        super(LuhnClassifier, self).__init__()
        self.L = length
        self.batch_size = batch_size

        self.neural_model = DigitClassifier(batch_size)
        self.luhn_model = LuhnModel()

    def call(self, inputs, training=None, mask=None):
        x = tf.concat([inputs[:, i, ...] for i in range(self.L)], axis=0)
        x = self.neural_model(x)
        x = [
            x[i * self.batch_size : (i + 1) * self.batch_size, :] for i in range(self.L)
        ]
        c = [construct_pint(x[i], 0) for i in range(self.L)]
        return self.luhn_model(c)


class LuhnModel(tf.keras.Model):

    def __init__(self):
        super(LuhnModel, self).__init__()

    def call(self, inputs, training=None, mask=None):
        check_digit = inputs[0]
        check = construct_pint(tf.convert_to_tensor([0.0]), lower=0)

        for i, digit in enumerate(inputs):
            if i % 2 == len(inputs) % 2:
                check = ifthenelse(
                    digit,
                    lt=5,
                    tbranch=lambda x: 2 * x,
                    fbranch=lambda x: 2 * x - 9,
                    accumulate=check,
                )
            else:
                check = check + digit
            check = check % 10
        check_value = check
        return log_expectation(check_digit + check_value == 10)
