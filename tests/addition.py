import os
import numpy as np
import tensorflow as tf

from plia import construct_pint


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

DIGITS = 2

probs = np.ones([10, 10]) * 0.1
probs[:, 0] = 0.99
probs[:, 1:] = 0.01 / 9

number1 = construct_pint(probs, 0, 9)
number2 = construct_pint(probs, 0, 9)
for _ in range(DIGITS - 1):
    number1 = number1 * 10
    number2 = number2 * 10
    number1 = number1 + construct_pint(probs, 0, 9)
    number2 = number2 + construct_pint(probs, 0, 9)

sum = number1 + number2

print(tf.reduce_sum(sum.probs, -1))
print(sum.E)
