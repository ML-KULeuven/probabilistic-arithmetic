import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from experiments.luhn.data.generation import luhn_checksum as lc
from plia import PInt, ifthenelse, log_expectation


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def luhn(identifier):
    check_digit = identifier[0]
    check_value = luhn_checksum(identifier[1:])
    return log_expectation(check_digit + check_value == 10)


def luhn_checksum(identifier):
    check = PInt(tf.convert_to_tensor([0.0]), lower=0)

    for i, digit in enumerate(identifier):
        if i % 2 == len(identifier) % 2:
            digit = ifthenelse(
                digit,
                lt=5,
                tbranch=lambda x: 2 * x,
                fbranch=lambda x: 2 * x - 9,
            )
            check = check + digit
        else:
            check = check + digit
        check = check % 10
    return check


if __name__ == "__main__":
    length = 2
    probs = np.ones([1, 10])
    probs[:, 3] = 0.99
    probs[:, 0:3] = 0.01 / 9
    probs[:, 4:] = 0.01 / 9
    probs = tf.constant(probs, dtype=tf.float32)
    identifier = [PInt(probs, lower=0, log_input=False) for _ in range(length)]
    constraint = luhn_checksum(identifier)
    true_id = [tf.argmax(x.logits, axis=-1) for x in identifier]
    print(true_id)
    check = lc(tf.concat(true_id, axis=-1))
    print(tf.exp(constraint.logits))
    print(check)
