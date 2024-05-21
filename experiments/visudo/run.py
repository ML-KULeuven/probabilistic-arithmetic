import os
import sys
import wandb
import tensorflow as tf
import multiprocessing as mp
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PARENT_DIR / "../.."))

from classifier import ViSudoClassifier
from data.generation import create_loader
from trainer import Trainer
from argparse import ArgumentParser
from evaluate import sudoku_accuracy


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def train(grid_size, learning_rate, batch_size, N_epochs, seed):
    wandb.init(
        project="probabilistic-arithmetic",
        name=f"visudo_{grid_size}_{seed}",
        config={
            "grid_size": grid_size,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": N_epochs,
            "seed": seed,
        },
        mode="disabled",
    )
    model = ViSudoClassifier(grid_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    train_data, val_data, test_data = create_loader(grid_size, batch_size)

    trainer = Trainer(
        model,
        optimizer,
        loss_object,
        train_data,
        val_data,
        sudoku_accuracy,
        epochs=N_epochs,
        log_its=10,
    )
    trainer.train()

    test_accuracy = sudoku_accuracy(model, test_data)

    print(f"Test accuracy: {test_accuracy}")

    wandb.log({"test_accuracy": test_accuracy})
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--N_epochs", type=int, default=2000)
    parser.add_argument("--N_runs", type=int, default=1)
    parser.add_argument("--N_workers", type=int, default=1)
    args = parser.parse_args()

    multiprocess_runs = args.N_runs // args.N_workers

    for seed in range(multiprocess_runs):
        p = mp.Pool(args.N_workers)
        p.starmap(
            train,
            [
                (
                    args.grid_size,
                    args.learning_rate,
                    args.batch_size,
                    args.N_epochs,
                    args.N_workers * seed + i,
                )
                for i in range(args.N_workers)
            ],
        )
    p.close()
    p.join()
