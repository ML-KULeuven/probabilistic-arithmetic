import tensorflow as tf


def sum_accuracy(model, data):
    correct = 0
    total = 0
    for batch in data:
        images, label = batch
        prediction = model(images)
        prediction = tf.argmax(prediction.logits, axis=-1)
        correct += tf.reduce_sum(tf.cast(prediction == label, tf.int32))
        total += tf.size(label)
    return correct / total


def cary_sum_accuracy(model, data):
    correct = 0
    total = 0
    for batch in data:
        images, label = batch
        prediction = model(images)
        prediction = tf.stack([pred.logits for pred in prediction], axis=-1)
        prediction = tf.argmax(prediction, axis=-2)
        all_correct = tf.reduce_all(prediction == label, axis=-1)
        correct += tf.reduce_sum(tf.cast(all_correct, tf.int32))
        total += tf.size(label[:, 0])
    return correct / total
