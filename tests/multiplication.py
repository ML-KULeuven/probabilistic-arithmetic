import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append("..")


from plia import construct_pint


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def main():
    logits = tf.random.uniform((10,))
    x = construct_pint(logits, 0)
    const = 4

    product = x * const

    print(product)


if __name__ == "__main__":
    main()
