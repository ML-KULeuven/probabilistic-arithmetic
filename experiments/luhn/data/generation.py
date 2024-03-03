import os
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent


VAL_FRACTION = 0.02
TRAINVAL_SIZE = 60000
VAL_SIZE = int(TRAINVAL_SIZE * VAL_FRACTION)


def luhn_checlsum(identifier):
    check = 0
    for i in range(identifier.shape[0]):
        digit = identifier[i]
        if i % 2 == identifier.shape[0] % 2:
            check += 2 * digit
            if digit > 4:
                check -= 9
        else:
            check += digit
        check = check % 10
    return check


def check_identifier(identifier):
    check_digit = identifier[0]
    check_value = luhn_checlsum(identifier[1:])
    return check_digit + check_value == 10


def create_identifier(id_length, x, y):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    x = x[indices]
    y = y[indices]

    N_ids = x.shape[0] // id_length

    identifers = []
    labels = []

    for i in range(N_ids):
        identifer = x[i : i + id_length, ...]
        label = y[i : i + id_length]
        label = check_identifier(label)

        identifers.append(identifer)
        labels.append(label)

    return identifers, labels


def create_loader(batch_size, id_length):

    train_data_file = PARENT_DIR / "data" / f"{id_length}_train.pkl"
    val_data_file = PARENT_DIR / "data" / f"{id_length}_val.pkl"
    test_data_file = PARENT_DIR / "data" / f"{id_length}_test.pkl"

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

        train_data = create_identifier(id_length, x_train, y_train)
        val_data = create_identifier(id_length, x_val, y_val)
        test_data = create_identifier(id_length, x_test, y_test)

        pickle.dump(train_data, open(train_data_file, "wb+"))
        pickle.dump(val_data, open(val_data_file, "wb+"))
        pickle.dump(test_data, open(test_data_file, "wb+"))

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_data[0], train_data[1]))
        .shuffle(TRAINVAL_SIZE - VAL_SIZE)
        .batch(batch_size=batch_size, drop_remainder=True)
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data[0], val_data[1])).batch(
        batch_size
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_data[0], test_data[1])
    ).batch(batch_size, drop_remainder=True)

    return train_dataset, val_dataset, test_dataset
