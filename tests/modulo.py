import os
import sys
import numpy as np
import tensorflow as tf

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

sys.path.append("..")
from plia import construct_pint


def main():
    logits = tf.random.uniform((18,))
    x = construct_pint(logits, -1)
    const = 4

    mod = x % const

    print(logits)
    print(tf.exp(mod.logits))


if __name__ == "__main__":
    main()
