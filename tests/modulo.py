import os
import sys
import tensorflow as tf
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from plia import PInt

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def main():
    logits = tf.random.uniform((18,))
    x = PInt(logits, -1)
    const = 4

    mod = x % const

    print(logits)
    print(tf.exp(mod.logits))


if __name__ == "__main__":
    main()
