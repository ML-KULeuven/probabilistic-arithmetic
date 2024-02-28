import os

import numpy as np
import tensorflow as tf

from plia.tools import ifthenelse


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


ID_LENGTH = 5

probs = np.ones([10, 10]) * 0.1
probs[:, 0] = 0.99
probs[:, 1:] = 0.01 / 9


def checksum(identifier):
    check = np.array([-np.inf] * 9)
    check[4] = 0.0

    for i, digit in enumerate(identifier):
        if i % 2 == len(identifier) % 2:
            check = check + ifthenelse(
                digit, lambda x: x > 4, lambda x: 2 * x - 9, lambda x: 2 * x
            )
        else:
            check = check + digit

    return check
