import os
import subprocess
import zipfile

import numpy as np
import tensorflow as tf
import einops as E

from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent
MNIST_DIM = 28


def get_data_url(grid_size):
    return f"https://linqs-data.soe.ucsc.edu/public/datasets/ViSudo-PC/v01/ViSudo-PC_dimension::{grid_size}_datasets::mnist_strategy::simple.zip"


def get_zipfile_name(grid_size):
    return f"ViSudo-PC_dimension::{grid_size}_datasets::mnist_strategy::simple"


def get_unzipped_dir(grid_size, unzip_root):
    zipfile_name = get_zipfile_name(grid_size)
    return unzip_root / "tmp" / "ViSudo-PC" / zipfile_name


def download_and_unzip(grid_size, output_dir):
    data_url = get_data_url(grid_size)
    zipfile_name = get_zipfile_name(grid_size)

    unzipped_dir = get_unzipped_dir(grid_size, output_dir)

    if not os.path.exists(output_dir / f"{zipfile_name}.zip"):
        download_with_wget(data_url, output_dir)
    if not os.path.isdir(unzipped_dir):
        unzip_file(output_dir / f"{zipfile_name}.zip", output_dir)


def download_with_wget(url, output_dir):
    try:
        subprocess.run(["wget", url, "-P", output_dir], check=True)
        print("Download successful!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")


def unzip_file(zip_file, output_dir):
    try:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        print("Unzipped successfully!")
    except Exception as e:
        print(f"Error unzipping file: {e}")


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

    unzip_dir = get_unzipped_dir(grid_size, PARENT_DIR)
    data_dir = (
        unzip_dir
        / f"dimension::{grid_size}"
        / "datasets::mnist/strategy::simple"
        / "strategy::simple"
        / f"numTrain::{num_train}"
        / "numTest::00100"
        / "numValid::00100"
        / "corruptChance::0.50"
        / f"overlap::{overlap}"
        / f"split::{spl}"
    )

    grids_file = data_dir / f"{partition}_puzzle_pixels.txt"
    labels_file = data_dir / f"{partition}_puzzle_labels.txt"

    labels = np.loadtxt(labels_file, delimiter="\t", dtype=int)
    labels = labels[:, 0]

    grids = np.loadtxt(grids_file, delimiter="\t", dtype=float)

    if not use_negative:
        indices = np.where(labels == 1)[0]
        grids = grids[indices]
        labels = labels[indices]

    grids = E.rearrange(
        grids,
        f"b (gs1 gs2 md1 md2) -> b gs1 gs2 md1 md2",
        gs1=grid_size,
        gs2=grid_size,
        md1=MNIST_DIM,
        md2=MNIST_DIM,
    )

    return grids, labels


def create_loader(
    grid_size: int,
    batch_size: int = 10,
    num_train: str = "00100",
    overlap: str = "0.00",
    split: int = 1,
    use_negative: bool = False,
):

    download_and_unzip(grid_size, PARENT_DIR)

    train_data = load_visudo(
        grid_size, "train", num_train, overlap, split, use_negative
    )
    val_data = load_visudo(grid_size, "valid", num_train, overlap, split, True)
    test_data = load_visudo(grid_size, "test", num_train, overlap, split, True)

    TRAIN_BUF = len(train_data[0])

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_data[0], train_data[1]))
        .shuffle(TRAIN_BUF)
        .batch(batch_size=batch_size, drop_remainder=True)
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data[0], val_data[1])).batch(
        batch_size=batch_size, drop_remainder=True
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_data[0], test_data[1])
    ).batch(50)

    return train_dataset, val_dataset, test_dataset
