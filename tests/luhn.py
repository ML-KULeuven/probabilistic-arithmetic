import os

import numpy as np
import tensorflow as tf

from plia import construct_pint, ifthenelse, log_expectation


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def luhn(identifier):
    check_digit = identifier[0]
    check_value = luhn_checksum(identifier[1:])
    return log_expectation(check_digit - check_value == 10)


def luhn_checksum(identifier):
    check = construct_pint(tf.convert_to_tensor([0]), lower=0)

    for i, digit in enumerate(identifier):
        if i % 2 == len(identifier) % 2:
            check = ifthenelse(
                digit,
                lt=5,
                tbranch=lambda x: 2 * x,
                fbranch=lambda x: 2 * x - 9,
                accumulate=check,
            )
        else:
            check = check + digit
        check = check % 10

    return check
