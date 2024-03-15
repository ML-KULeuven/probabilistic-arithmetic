import os
import pickle
import torch
import numpy as np
import tensorflow as tf

from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent
MNIST_DIM = 28


def convertToInts(dataset: np.array):
    values = list(sorted(set(dataset.flatten().tolist())))

    valueMap = {}
    for value in values:
        valueMap[value] = len(valueMap)

    trainOut = []

    # for (outData, inData) in [[trainOut, train], [testOut, test]]:
    for row in dataset:
        trainOut.append([valueMap[value] for value in row])

    train = np.stack(trainOut)

    return train


def load_visudo(
    grid_size: int,
    partition: str,
    num_train: str,
    overlap: str,
    split: int,
    use_negative: bool,
):
    spl = str(split)
    if split < 10:
        spl = "0" + spl
    data_dir = (
        PARENT_DIR
        / "ViSudo-PC"
        / f"ViSudo-PC_dimension::{grid_size}_datasets::mnist_strategy::simple/"
        f"dimension::{grid_size}/datasets::mnist/strategy::simple/strategy::simple/"
        f"numTrain::{num_train}/numTest::00100/numValid::00100/corruptChance::0.50/"
        f"overlap::{overlap}/split::{spl}"
    )

    grids_file = os.path.join(data_dir, f"{partition}_puzzle_pixels.txt")

    labels_file = os.path.join(data_dir, f"{partition}_puzzle_labels.txt")
    labels = np.loadtxt(labels_file, delimiter="\t", dtype=str)

    labels = convertToInts(labels)
    labels = 1 - torch.max(torch.tensor(labels), dim=1)[1]

    grids = np.loadtxt(grids_file, delimiter="\t", dtype=float)
    grids = torch.tensor(grids, dtype=torch.float32)

    if not use_negative:
        grids = grids[labels == 1]
        labels = labels[labels == 1]

    grids = tf.constant(grids)
    grids = tf.reshape(grids, (-1, grid_size, grid_size, MNIST_DIM, MNIST_DIM))

    labels = tf.constant(labels)

    return grids, labels


def create_loader(
    grid_size: int,
    batch_size: int = 10,
    num_train: str = "00100",
    overlap: str = "0.00",
    split: int = 1,
    use_negative: bool = False,
):
    train_data_file = PARENT_DIR / "data" / f"{grid_size}_train_{int(use_negative)}.pkl"
    val_data_file = PARENT_DIR / "data" / f"{grid_size}_val_{int(use_negative)}.pkl"
    test_data_file = PARENT_DIR / "data" / f"{grid_size}_test_{int(use_negative)}.pkl"

    if os.path.exists(train_data_file):
        train_data = pickle.load(open(train_data_file, "rb"))
        val_data = pickle.load(open(val_data_file, "rb"))
        test_data = pickle.load(open(test_data_file, "rb"))
    else:
        train_data = load_visudo(
            grid_size, "train", num_train, overlap, split, use_negative
        )
        val_data = load_visudo(grid_size, "valid", num_train, overlap, split, True)
        test_data = load_visudo(grid_size, "test", num_train, overlap, split, True)

        if not os.path.exists(PARENT_DIR / "data"):
            os.makedirs(PARENT_DIR / "data")
        pickle.dump(train_data, open(train_data_file, "wb+"))
        pickle.dump(val_data, open(val_data_file, "wb+"))
        pickle.dump(test_data, open(test_data_file, "wb+"))

    TRAIN_BUF = len(train_data[0])
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_data[0], train_data[1]))
        .shuffle(TRAIN_BUF)
        .batch(batch_size=batch_size, drop_remainder=True)
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data[0], val_data[1])).batch(
        50
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_data[0], test_data[1])
    ).batch(50, drop_remainder=True)

    return train_dataset, val_dataset, test_dataset
