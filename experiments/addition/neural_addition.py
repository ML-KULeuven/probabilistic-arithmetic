import tensorflow as tf
import wandb
import os
import multiprocessing as mp

from sum_classifier import SumClassifier
from data.generation import create_loader
from trainer import Trainer
from evaluate import sum_accuracy


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

DIGITS = 6
LR = 1e-3
NR_RUNS = 10
EPOCHS = 20


def train(seed):
    wandb.init(
        project="probabilistic-arithmetic",
        name=f"addition_{DIGITS}_{seed}",
        config={"digits": DIGITS, "lr": LR, "seed": seed, "epochs": EPOCHS},
    )

    model = SumClassifier(DIGITS)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_data, val_data, test_data = create_loader(DIGITS)

    trainer = Trainer(
        model, optimizer, loss_object, train_data, val_data, epochs=EPOCHS
    )
    trainer.train()

    test_accuracy = sum_accuracy(model, test_data)
    wandb.log({"test_accuracy": test_accuracy.numpy()})
    print(f"Test accuracy: {test_accuracy.numpy()}")


for seed in range(NR_RUNS):
    p = mp.Process(target=train, args=(seed,))
    p.start()
    p.join()
