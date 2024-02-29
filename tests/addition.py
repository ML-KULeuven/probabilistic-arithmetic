import os
import numpy as np
import tensorflow as tf

from plia import construct_pint, log_expectation


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

DIGITS = 2

probs = np.ones([10, 10]) * 0.1
probs[:, 0] = 0.99
probs[:, 1:] = 0.01 / 9
probs = tf.constant(probs, dtype=tf.float32)

number1 = construct_pint(probs, 0)
number2 = construct_pint(probs, 0)
for _ in range(DIGITS - 1):
    number1 = number1 * 10
    number2 = number2 * 10
    number1 = number1 + construct_pint(probs, 0)
    number2 = number2 + construct_pint(probs, 0)

sum = number1 + number2

print(sum)
# print(tf.exp(log_expectation(sum)))
