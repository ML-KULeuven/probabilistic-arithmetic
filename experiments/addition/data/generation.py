import os
import pickle
import tensorflow as tf

from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent


TRAINVAL_SIZE = 60000
VAL_SIZE = 1000
TEST_SIZE = 10000


def sum_labels(labels):
    value = 0
    for i in range(1, len(labels) + 1):
        value += tf.cast(labels[-i], dtype=tf.int64) * 10 ** (i - 1)
    return value


def create_numbers(digits_per_number, numbers, data_x, data_y, batch_size=10):
    images = []
    labels = []

    data_size = (
        data_x.shape[0] // (numbers * digits_per_number * batch_size) * (batch_size)
    )
    for i in range(data_size):
        number_images = []
        for j in range(numbers):
            number_images.append(
                data_x[
                    i * numbers * digits_per_number
                    + j * digits_per_number : i * numbers * digits_per_number
                    + (j + 1) * digits_per_number,
                    ...,
                ]
            )
        number_images = tf.stack(number_images, axis=0)

        number_sums = []
        for j in range(numbers):
            number_labels = data_y[
                i * numbers * digits_per_number
                + j * digits_per_number : i * numbers * digits_per_number
                + (j + 1) * digits_per_number
            ]
            number_sums.append(sum_labels(number_labels))

        label = sum(number_sums)

        images.append(number_images)
        labels.append(label)

    return images, labels


def create_loader(
    digits_per_number: int,
    numbers: int = 2,
    batch_size: int = 10,
):

    train_data_file = PARENT_DIR / "data" / f"{digits_per_number}_train.pkl"
    val_data_file = PARENT_DIR / "data" / f"{digits_per_number}_val.pkl"
    test_data_file = PARENT_DIR / "data" / f"{digits_per_number}_test.pkl"

    if os.path.exists(train_data_file):
        train_data = pickle.load(open(train_data_file, "rb"))
        val_data = pickle.load(open(val_data_file, "rb"))
        test_data = pickle.load(open(test_data_file, "rb"))
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype("float32") / 255.0  # [60000, 28, 28]
        x_test = x_test.astype("float32") / 255.0  # [10000, 28, 28]

        x_train = x_train[:-VAL_SIZE, ...]
        y_train = y_train[:-VAL_SIZE]

        x_val = x_train[-VAL_SIZE:, ...]
        y_val = y_train[-VAL_SIZE:]

        train_data = create_numbers(digits_per_number, numbers, x_train, y_train)
        val_data = create_numbers(digits_per_number, numbers, x_val, y_val)
        test_data = create_numbers(digits_per_number, numbers, x_test, y_test)

        if not os.path.exists(PARENT_DIR / "data"):
            os.makedirs(PARENT_DIR / "data")
        pickle.dump(train_data, open(train_data_file, "wb+"))
        pickle.dump(val_data, open(val_data_file, "wb+"))
        pickle.dump(test_data, open(test_data_file, "wb+"))

    TRAIN_BUF = (
        (TRAINVAL_SIZE - VAL_SIZE)
        // (numbers * digits_per_number * batch_size)
        * (numbers * digits_per_number * batch_size)
    )

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_data[0], train_data[1]))
        .shuffle(TRAIN_BUF)
        .batch(batch_size=batch_size, drop_remainder=True)
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data[0], val_data[1])).batch(
        batch_size
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_data[0], test_data[1])
    ).batch(batch_size, drop_remainder=True)

    return train_dataset, val_dataset, test_dataset
