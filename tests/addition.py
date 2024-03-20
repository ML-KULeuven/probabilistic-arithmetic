import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))


from plia import PInt, log_expectation

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def main():
    DIGITS = 3

    probs = np.ones([10, 10]) / 10
    # probs[:, 0] = 0.99
    # probs[:, 1:] = 0.01 / 9
    probs = tf.constant(probs, dtype=tf.float32)

    number1 = PInt(probs, 0, log_input=False)
    number2 = PInt(probs, 0, log_input=False)
    for _ in range(DIGITS - 1):
        number1 = number1 * 10
        number2 = number2 * 10
        number1 = number1 + PInt(probs, 0, log_input=False)
        number2 = number2 + PInt(probs, 0, log_input=False)

    sum = number1 + number2

    print(sum)
    print(tf.exp(log_expectation(sum)))


if __name__ == "__main__":
    main()
