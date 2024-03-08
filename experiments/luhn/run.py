import os
import sys
import multiprocessing as mp
import argparse
import tensorflow as tf
import wandb
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PARENT_DIR / "../.."))
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from classifier import LuhnClassifier, LuhnCheckClassifier
from data.generation import create_loader
from trainer import Trainer
from experiments.luhn.weighted_ce import WeightedCE
from evaluate import weighted_accuracy, accuracy, luhn_accuracy


def train(id_length, learning_rate, batch_size, N_epochs, seed):
    wandb.init(
        project="probabilistic-arithmetic",
        name=f"luhn_{id_length}_{seed}",
        config={
            "id_length": id_length,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "N_epochs": N_epochs,
            "seed": seed,
        },
    )

    model = LuhnCheckClassifier()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # loss_object = WeightedCE(weight=1.0 + tf.math.log(9.0))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_data, val_data, test_data = create_loader(batch_size, id_length)

    trainer = Trainer(
        model,
        optimizer,
        loss_object,
        train_data,
        val_data,
        luhn_accuracy,
        epochs=N_epochs,
    )
    trainer.train()

    test_accuracy = luhn_accuracy(model, test_data)
    wandb.log({"test_accuracy": test_accuracy.numpy()})
    print(f"Test accuracy: {test_accuracy.numpy()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_length", default=2, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--N_epochs", default=10, type=int)
    parser.add_argument("--N_runs", default=1, type=int)

    args = parser.parse_args()

    for seed in range(args.N_runs):
        p = mp.Process(
            target=train,
            args=(
                args.id_length,
                args.learning_rate,
                args.batch_size,
                args.N_epochs,
                seed,
            ),
        )
        p.start()
        p.join()
