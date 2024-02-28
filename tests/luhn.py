import os
import functools

import numpy as np
import tensorflow as tf

from plia import PInt, ifthenelse


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def tbranch(check, x):
    return check + 2 * x


def fbranch(check, x):
    return tbranch(check, x) - 9


def checksum(identifier):
    check = PInt(tf.convert_to_tensor([-np.inf]), lower=0)

    for i, digit in enumerate(identifier):
        if i % 2 == len(identifier) % 2:
            tb = functools.partial(tbranch, check=check)
            fb = functools.partial(fbranch, check=check)

            check = ifthenelse(digit, lt=5, tbranch=tb, fbranch=fb)
        else:
            check = check + digit

    return check
