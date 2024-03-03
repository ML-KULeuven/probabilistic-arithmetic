import tensorflow as tf
import einops as E

from plia import construct_pint, ifthenelse, log_expectation


class LuhnClassifier(tf.keras.Model):

    def __init__(self):
        super(LuhnClassifier, self).__init__()

        self.classifier = DigitClassifier()

    def call(self, inputs, training=None, mask=None):
        x = self.classifier(inputs)
        x = self.check_identifer(x)
        return x

    def check_identifer(self, x):
        identifier = [construct_pint(x[:, i, :], 0) for i in range(x.shape[1])]

        check_digit = identifier[0]
        check_value = self.luhn_checksum(identifier[1:])
        return log_expectation(check_digit + check_value == 10)

    def luhn_checksum(self, identifier):

        b = identifier[0].logits.shape[0]

        check = construct_pint(tf.constant(0.0, shape=(b, 1)), lower=0)
        for i, digit in enumerate(identifier):
            if i % 2 == len(identifier) % 2:
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
        return check


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
        b = inputs.shape[0]
        inputs = E.rearrange(inputs, "b i ... -> (b i) ...")
        inputs = tf.expand_dims(inputs, axis=-1)
        x = self.model(inputs)
        x = E.rearrange(x, "(b i) ... -> b i ...", b=b)
        return x
