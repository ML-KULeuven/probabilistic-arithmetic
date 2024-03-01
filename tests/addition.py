import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append("..")


from plia import construct_pint


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

DIGITS = 2

probs = np.ones([10, 10]) * 0.1
probs[:, 0] = 0.99
probs[:, 1:] = 0.01 / 9
probs = tf.constant(probs, dtype=tf.float32)

number1 = construct_pint(probs, 0, log_input=False)
number2 = construct_pint(probs, 0, log_input=False)
for _ in range(DIGITS - 1):
    number1 = number1 * 10
    number2 = number2 * 10
    number1 = number1 + construct_pint(probs, 0, log_input=False)
    number2 = number2 + construct_pint(probs, 0, log_input=False)

sum = number1 + number2

print(sum)
# print(tf.exp(log_expectation(sum)))
