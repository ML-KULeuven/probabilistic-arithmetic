import os
import sys
import time
import argparse
from pathlib import Path
import yaml

PARENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PARENT_DIR / "../.."))
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


import tensorflow as tf

GPUS = tf.config.experimental.list_physical_devices("GPU")

from plia import construct_pint, log_expectation, ifthenelse

PROBLEMS = ["sum", "le", "eq", "luhn"]


class Timer:
    def __init__(self, times, bitwidth):
        self.times = times
        self.bitwidth = bitwidth

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args):
        t = time.time() - self.start_time
        self.times.append(t)
        print("%s: %.4fs (bitwidth: %i)" % ("time", t, self.bitwidth))


def luhn(*identifier):
    check_digit = identifier[0]
    check_value = luhn_checksum(identifier[1:])
    return check_digit + check_value == 10


def luhn_checksum(identifier):
    check = construct_pint(tf.convert_to_tensor([0.0]), lower=0)

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


str2func = {
    "sum": lambda *args: args[0] + args[1],
    "le": lambda *args: args[0] <= args[1],
    "eq": lambda *args: args[0] == args[1],
    "luhn": lambda *args: luhn(*args),
}


def make_path(device, problem):
    path = PARENT_DIR / Path("results") / Path(f"{device}") / Path(f"{problem}")

    if not os.path.exists(path):
        os.makedirs(path)
    return path


def run_expecation(problem, max_bitwidth, device):
    """Start with a dry run to not time TF backend initialisation cost"""
    bitwidth = 1
    number1 = construct_pint(tf.random.uniform((2**bitwidth,), minval=0, maxval=1), 0)
    number2 = construct_pint(tf.random.uniform((2**bitwidth,), minval=0, maxval=1), 0)
    number1 + number2
    result = str2func[problem](number1, number2)
    result = log_expectation(result)

    print(f"\ndevice: {device}, problem: {problem}")

    times = []
    for bitwidth in range(1, max_bitwidth):

        """
        Watch out in this experiment! Tensorflow compiles and caches some operations in the background!
        For example, since the addition is used in the comparisons, it will use the cached operation from the first run!

        Run all of these both separately and when compiled?

        Additional info: bitwidth of 24 uses +- 15GB of VRAM and seems to be max for GPU
        """

        if problem == "luhn":
            fargs = [
                construct_pint(tf.random.uniform((10,)), 0) for _ in range(bitwidth)
            ]
        else:
            number1 = construct_pint(tf.random.uniform((2**bitwidth,)), 0)
            number2 = construct_pint(tf.random.uniform((2**bitwidth,)), 0)
            fargs = (number1, number2)

        with Timer(times, bitwidth):
            result = str2func[problem](*fargs)
            result = log_expectation(result)
            tf.test.experimental.sync_devices()

    with open(make_path(device, problem) / "times.yaml", "w+") as f:
        yaml.dump(times, f, default_flow_style=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--max_bitwidth", default=24, type=int)

    args = parser.parse_args()

    for p in PROBLEMS:
        if GPUS and args.device == "gpu":
            tf.config.experimental.set_visible_devices(GPUS[0], "GPU")
        else:
            tf.config.experimental.set_visible_devices([], "GPU")
        run_expecation(p, args.max_bitwidth, args.device)
