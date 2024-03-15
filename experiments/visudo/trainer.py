import time
import wandb
import tensorflow as tf


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        loss_object,
        train_dataset,
        val_dataset,
        val_fn,
        epochs=10,
        log_its=100,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val_fn = val_fn
        self.epochs = epochs
        self.log_its = log_its

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            # loss = self.loss_object(label, predictions)
            loss = -tf.reduce_mean(predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    @tf.function
    def val_step(self, images, labels):
        predictions = self.model(images)
        # loss = self.loss_object(labels, predictions)
        loss = -tf.reduce_mean(predictions)
        return loss

    def evaluate(self):
        val_loss = tf.keras.metrics.Mean()
        for batch in self.val_dataset:
            images = batch[0]
            labels = batch[1]
            loss = self.val_step(images, labels)
            val_loss.update_state(loss)
        return val_loss.result().numpy()

    def train(self):
        avg_loss = tf.keras.metrics.Mean()
        duration = tf.keras.metrics.Sum()
        count = 0
        for epoch in range(self.epochs):
            for data in self.train_dataset:
                images = data[0]
                labels = data[1]
                start_time = time.time()
                loss = self.train_step(images, labels)
                avg_loss.update_state(loss)
                duration.update_state(time.time() - start_time)
                if count % self.log_its == 0:
                    acc = self.val_fn(self.model, self.val_dataset)
                    val_loss = self.evaluate()
                    data = [
                        epoch + 1,
                        count,
                        avg_loss.result().numpy(),
                        val_loss,
                        acc.numpy(),
                        duration.result().numpy(),
                    ]
                    print(
                        "Epoch {: >5}  Iteration: {: >5}   Loss: {: >20}    Val Loss: {: >20}    Accuracy: {: >8}  Time(s): {: >20} ".format(
                            *data
                        )
                    )
                    wandb.log(
                        {
                            "loss": avg_loss.result().numpy(),
                            "val_loss": val_loss,
                            "accuracy": acc.numpy(),
                            "time": duration.result().numpy(),
                        }
                    )
                    avg_loss.reset_states()
                    duration.reset_states()
                count += 1
