import tensorflow as tf


def sum_accuracy(model, data):
    correct = 0
    total = 0
    for batch in data:
        n1, n2, target = batch[:3]
        prediction = model([n1, n2])
        prediction = tf.argmax(prediction.logprobs, axis=-1)
        correct += tf.reduce_sum(tf.cast(prediction == target, tf.int32))
        total += tf.size(target)
    return correct / total