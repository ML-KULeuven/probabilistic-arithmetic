import tensorflow as tf

from algebra import Categorical


def is_one(probs):
    r = tf.reduce_sum(probs, axis=-1)
    return tf.experimental.numpy.isclose(r, 1.0)


def main():
    logits11 = tf.random.normal(shape=(1, 10))
    logits12 = tf.random.normal(shape=(1, 10))

    rv11 = Categorical(logits=logits11, lower=0)
    rv12 = Categorical(logits=logits12, lower=0)

    sum1 = rv11 + rv12
    print()
    print(rv11)
    print(rv12)
    print(sum1)
    sum1.print_probs()
    print("FAIL") if not is_one(sum1.probs) else print("PASS")

    ######################################################################
    ######################################################################
    ######################################################################

    logits21 = tf.random.normal(shape=(1, 13))
    logits22 = tf.random.normal(shape=(1, 10))

    rv21 = Categorical(logits=logits21, lower=4)
    rv22 = Categorical(logits=logits22, lower=-11)

    sum2 = rv21 + rv22
    print()
    print(rv21)
    print(rv22)
    print(sum2)
    sum2.print_probs()
    print("FAIL") if not is_one(sum2.probs) else print("PASS")


if __name__ == "__main__":
    main()
