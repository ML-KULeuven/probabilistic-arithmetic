import tensorflow as tf
import numpy as np
import einops as E

from plia import Krat, log_expectation, log1mexp


class ViSudoClassifier(tf.keras.Model):

    def __init__(self, grid_size: int = 9):
        super().__init__()
        self.digit_classifier = ViSudoDigitClassifier(grid_size)
        self.sudoku_solver = SudokuSolver(grid_size)

    def call(self, inputs, training=None, mask=None):
        x = self.digit_classifier(inputs)
        x = self.sudoku_solver(x)
        x = tf.reduce_sum(x, axis=-1)
        return x


class SudokuSolver(tf.keras.Model):

    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size

    def binarize(self, probs):
        # neg_probs = log1mexp(probs)
        neg_probs = -probs
        return tf.stack([neg_probs, probs], -1)
        # return E.rearrange([neg_probs, probs], "2 ... -> ... 2")

    def distinct_row_elements(self, inputs):
        return E.rearrange(inputs, "b r c p 2 -> b c r p 2")

    def distinct_column_elements(self, inputs):
        return E.rearrange(inputs, "b r c p 2 -> b c r p 2")

    def distinct_box_elements(self, inputs):
        box_dim = int(np.sqrt(self.grid_size))
        return E.rearrange(
            inputs,
            "b (r box_r) (c box_c) p -> b (r c) (box_r box_c) p",
            r=box_dim,
            c=box_dim,
        )

    def get_constraints(self, x, ctype):
        if ctype == "row":
            return E.rearrange(x, "b r c p binaries -> b (r p) c binaries")
        elif ctype == "column":
            return E.rearrange(x, "b r c p binaries -> b (c p) r binaries")
        elif ctype == "box":
            box_dim = int(np.sqrt(self.grid_size))
            return E.rearrange(
                x,
                "b (r box_r) (c box_c) p binaries -> b (r c p) (box_r box_c) binaries",
                r=box_dim,
                c=box_dim,
            )
        else:
            raise NotImplementedError()

    def call(self, inputs, training=None, mask=None):
        x = self.binarize(inputs)
        constraints = []

        constraints.append(self.get_constraints(x, "row"))
        constraints.append(self.get_constraints(x, "column"))
        if self.grid_size == 9:
            constraints.append(self.get_constraints(x, "box"))
        constraints = E.rearrange(
            constraints,
            "i b constraint_index constraints binaries -> b (constraint_index i) constraints binaries",
        )

        krat_constraints = Krat(constraints, 0)
        pintjes = krat_constraints.sum_reduce()
        expectation = log_expectation(pintjes == 1)
        return expectation


@tf.keras.utils.register_keras_serializable(package="Custom", name="entropy")
class EntropyRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, coefficient: float = 0.01):
        self.coefficient = coefficient

    def __call__(self, logits):
        return self.coefficient * tf.reduce_sum(tf.exp(logits) * logits)

    def get_config(self):
        return {"coefficient": self.coefficient}


class ViSudoDigitClassifier(tf.keras.Model):

    def __init__(self, grid_size: int = 9, entropy_regularizer: float = 1.0):
        self.grid_size = grid_size
        super(ViSudoDigitClassifier, self).__init__()

        entropy_regularizer = EntropyRegularizer(entropy_regularizer)

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(6, 5, activation="relu"))
        self.model.add(tf.keras.layers.MaxPooling2D())
        self.model.add(tf.keras.layers.Conv2D(16, 5, activation="relu"))
        self.model.add(tf.keras.layers.MaxPooling2D())
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(120, activation="relu"))
        self.model.add(tf.keras.layers.Dense(84, activation="relu"))
        self.model.add(
            tf.keras.layers.Dense(grid_size, activity_regularizer=entropy_regularizer)
        )

    def call(self, inputs, training=None, mask=None):
        inputs = E.rearrange(inputs, "b row column ... -> (b row column) ...")
        inputs = tf.expand_dims(inputs, axis=-1)
        x = self.model(inputs)
        x = E.rearrange(
            x,
            "(b row column) ... -> b row column ...",
            row=self.grid_size,
            column=self.grid_size,
        )
        return x
