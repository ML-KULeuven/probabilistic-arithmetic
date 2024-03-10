import os
import sys
import tensorflow as tf
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))


from plia import PInt


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def main():
    logits = tf.random.uniform((10,))
    x = PInt(logits, 0)
    const = 4

    product = x * const

    print(product)


if __name__ == "__main__":
    main()
