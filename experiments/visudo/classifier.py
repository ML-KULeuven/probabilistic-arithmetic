import tensorflow as tf
import numpy as np
import einops as E

from plia import PInt, log_expectation


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

    def binary_representation(self, inputs, grid_size):
        rows = []
        for i in range(grid_size):
            column = []
            for j in range(grid_size):
                binaries = []
                for k in range(grid_size):
                    logit_k = inputs[:, i, j, k]
                    binary = PInt(tf.stack([-logit_k, logit_k], -1), 0)
                    binaries.append(binary)
                column.append(binaries)
            rows.append(column)
        return rows

    def binary_numpy_representation(self, inputs, grid_size):
        representation = self.binary_representation(inputs, grid_size)
        representation = np.array(representation, dtype=object)
        return representation

    def distinct_numpy_row_elements(self, inputs):
        x = E.reduce(inputs, "row column binaries -> (row binaries)", "sum")
        row_constraints = [0] * len(x)
        for i, _ in enumerate(x):
            row_constraints[i] = log_expectation(x[i] == 1)
        row_constraints = tf.stack(row_constraints, axis=1)
        return row_constraints

    def distinct_row_elements(self, inputs):
        row_constraints = []
        for i, row in enumerate(inputs):
            row_constraints.append([0] * self.grid_size)
            for column in row:
                for k, binary in enumerate(column):
                    row_constraints[i][k] += binary
        for i, row in enumerate(row_constraints):
            for k, binary in enumerate(row):
                row_constraints[i][k] = log_expectation(row_constraints[i][k] == 1)
            row_constraints[i] = tf.stack(row_constraints[i], axis=1)
        row_constraints = tf.stack(row_constraints, axis=1)
        row_constraints = E.rearrange(
            row_constraints, "b row binaries -> b (row binaries)"
        )
        return row_constraints

    def distinct_numpy_column_elements(self, inputs):
        x = E.reduce(inputs, "row column binaries -> (column binaries)", "sum")
        column_constraints = [0] * len(x)
        for j, _ in enumerate(x):
            column_constraints[j] = log_expectation(x[j] == 1)
        column_constraints = tf.stack(column_constraints, axis=1)
        return column_constraints

    def distinct_column_elements(self, inputs):
        column_constraints = []
        for j in range(self.grid_size):
            column_constraints.append([0] * self.grid_size)
            for i in range(self.grid_size):
                for k in range(self.grid_size):
                    column_constraints[j][k] += inputs[i][j][k]
        for j, column in enumerate(column_constraints):
            for k, binary in enumerate(column):
                column_constraints[j][k] = log_expectation(
                    column_constraints[j][k] == 1
                )
            column_constraints[j] = tf.stack(column_constraints[j], axis=1)
        column_constraints = tf.stack(column_constraints, axis=1)
        column_constraints = E.rearrange(
            column_constraints, "b column binaries -> b (column binaries)"
        )
        return column_constraints

    def distinct_numpy_box_elements(self, inputs):
        row = inputs.shape[0]
        column = inputs.shape[1]
        binaries = inputs.shape[2]
        box_row = row // 3
        box_column = column // 3
        x = E.rearrange(
            inputs,
            "(box_row r) (box_column c) binaries -> box_row box_column (r c) binaries",
            r=3,
            c=3,
            box_row=box_row,
            box_column=box_column,
        )
        x = E.reduce(
            x,
            "box_row box_column box binaries -> (box_row box_column binaries)",
            "sum",
        )

        box_constraints = [0] * len(x)
        for i, _ in enumerate(x):
            box_constraints[i] = log_expectation(x[i] == 1)
        box_constraints = tf.stack(box_constraints, axis=1)
        return box_constraints

    def distinct_box_elements(self, inputs):
        box_constraints = []
        for i in range(self.grid_size // 3):
            box_constraints.append([])
            for j in range(self.grid_size // 3):
                box_constraints[i].append([0] * self.grid_size)
                for box_row in range(self.grid_size // 3):
                    for box_column in range(self.grid_size // 3):
                        for k in range(self.grid_size):
                            box_constraints[i][j][k] += inputs[i * 3 + box_row][
                                j * 3 + box_column
                            ][k]

        for i, box_row in enumerate(box_constraints):
            for j, box_column in enumerate(box_row):
                for k, binary in enumerate(box_column):
                    box_constraints[i][j][k] = log_expectation(
                        box_constraints[i][j][k] == 1
                    )
                box_constraints[i][j] = tf.stack(box_constraints[i][j], axis=1)
            box_constraints[i] = tf.stack(box_constraints[i], axis=1)
        box_constraints = tf.stack(box_constraints, axis=1)
        box_constraints = E.rearrange(
            box_constraints,
            "b box_row box_column binaries -> b (box_row box_column binaries)",
        )
        return box_constraints

    def call(self, inputs, training=None, mask=None):
        grid_size = inputs.shape[-1]

        x = self.binary_numpy_representation(inputs, grid_size)
        row_constraint = self.distinct_numpy_row_elements(x)
        column_constraint = self.distinct_numpy_column_elements(x)

        if grid_size == 9:
            box_constraint = self.distinct_numpy_box_elements(x)
            return tf.concat(
                [row_constraint, column_constraint, box_constraint], axis=-1
            )
        else:
            return tf.concat([row_constraint, column_constraint], axis=-1)


class ViSudoDigitClassifier(tf.keras.Model):

    def __init__(self, grid_size: int = 9):
        super(ViSudoDigitClassifier, self).__init__()

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(6, 5, activation="relu"))
        self.model.add(tf.keras.layers.MaxPooling2D())
        self.model.add(tf.keras.layers.Conv2D(16, 5, activation="relu"))
        self.model.add(tf.keras.layers.MaxPooling2D())
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(120, activation="relu"))
        self.model.add(tf.keras.layers.Dense(84, activation="relu"))
        self.model.add(tf.keras.layers.Dense(grid_size))

    def call(self, inputs, training=None, mask=None):
        b, row, column = inputs.shape[0:3]
        inputs = E.rearrange(inputs, "b row column ... -> (b row column) ...")
        inputs = tf.expand_dims(inputs, axis=-1)
        x = self.model(inputs)
        x = E.rearrange(
            x, "(b row column) ... -> b row column ...", b=b, row=row, column=column
        )
        return x
