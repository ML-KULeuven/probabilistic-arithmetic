import tensorflow as tf
import wandb
import os
import multiprocessing as mp

from luhn_classifier import LuhnClassifier
from data.generation import create_loader
from addition.trainer import Trainer
from evaluate import luhn_accuracy


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

ID_LENGTH = 10
LR = 1e-3
NR_RUNS = 10
EPOCHS = 20


def train(seed):
    wandb.init(
        project="probabilistic-arithmetic",
        name=f"luhn_{ID_LENGTH}_{seed}",
        config={"id_length": ID_LENGTH, "lr": LR, "seed": seed, "epochs": EPOCHS},
    )

    model = LuhnClassifier(ID_LENGTH)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    train_data, val_data, test_data = create_loader(ID_LENGTH)

    trainer = Trainer(
        model,
        optimizer,
        loss_object,
        train_data,
        val_data,
        luhn_accuracy,
        epochs=EPOCHS,
    )
    trainer.train()

    test_accuracy = luhn_accuracy(model, test_data)
    wandb.log({"test_accuracy": test_accuracy.numpy()})
    print(f"Test accuracy: {test_accuracy.numpy()}")


for seed in range(NR_RUNS):
    p = mp.Process(target=train, args=(seed,))
    p.start()
    p.join()
