import tensorflow as tf


def accuracy(model, data):
    correct = 0
    total = 0
    for batch in data:
        identifier, label = batch
        prediction = model(identifier)
        prediction = prediction > tf.math.log(0.5)
        correct += tf.reduce_sum(tf.cast(prediction == label, tf.int32))
        total += tf.size(label)
    return correct / total


def weighted_accuracy(model, data):
    correct = 0
    total = 0
    for batch in data:
        identifier, label = batch
        prediction = model(identifier)
        prediction = prediction > tf.math.log(0.5)
        pos_id = tf.where(label)
        pos_correct = 0.9 * tf.reduce_sum(
            tf.where(
                tf.gather_nd(prediction, pos_id) == tf.gather_nd(label, pos_id),
                1.0,
                0.0,
            )
        )
        neg_id = tf.where(label == False)
        neg_correct = 0.1 * tf.reduce_sum(
            tf.where(
                tf.gather_nd(prediction, neg_id) == tf.gather_nd(label, neg_id),
                1.0,
                0.0,
            )
        )
        correct += pos_correct + neg_correct
        total += int(tf.size(pos_id)) * 0.9 + int(tf.size(neg_id)) * 0.1
    return correct / total


def luhn_accuracy(model, data):
    correct = 0
    total = 0
    for identifier, label in data:
        prediction = model(identifier)
        prediction = tf.argmax(prediction.logits, axis=-1, output_type=tf.int32)
        correct += tf.reduce_sum(tf.cast(prediction == label, tf.int32))
        total += tf.size(label)
    return correct / total
