import tensorflow as tf


def sudoku_accuracy(model, data):
    correct = 0
    total = 0
    for batch in data:
        visudo, label = batch
        prediction = model(visudo)
        prediction = tf.cast(tf.round(tf.exp(prediction)), tf.int64)
        correct += tf.reduce_sum(tf.cast(prediction == label, tf.int32))
        total += tf.size(label)
    return correct / total
