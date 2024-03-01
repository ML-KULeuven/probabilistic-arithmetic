import os
import sys
import time
import argparse
from pathlib import Path
import yaml

sys.path.append("../..")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


import tensorflow as tf

GPUS = tf.config.experimental.list_physical_devices("GPU")

from plia import construct_pint, log_expectation


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


str2func = {
    "sum": lambda x, y: x + y,
    "le": lambda x, y: x < y,
    "eq": lambda x, y: x == y,
}


def make_path(device, problem):
    path = (
        Path(__file__).resolve().parent
        / Path("results")
        / Path(f"{device}")
        / Path(f"{problem}")
    )

    if not os.path.exists(path):
        os.makedirs(path)
    return path


def run_comparison(device, problem, max_bitwidth):
    """Start with a dry run to not time TF backend initialisation cost"""
    bitwidth = 1
    number1 = construct_pint(tf.random.uniform((2**bitwidth,), minval=0, maxval=1), 0)
    number2 = construct_pint(tf.random.uniform((2**bitwidth,), minval=0, maxval=1), 0)
    number1 + number2
    result = str2func[problem](number1, number2)
    result = log_expectation(result)

    print(f"\ndevice: {device}, problem: {problem}")

    times = []
    for bitwidth in range(max_bitwidth):

        """
        Watch out in this experiment! Tensorflow compiles and caches some operations in the background!
        For example, since the addition is used in the comparisons, it will use the cached operation from the first run!

        Run all of these both separately and when compiled?

        Additional info: bitwidth of 24 uses +- 15GB of VRAM and seems to be max for GPU
        """

        number1 = construct_pint(tf.random.uniform((2**bitwidth,)), 0)
        number2 = construct_pint(tf.random.uniform((2**bitwidth,)), 0)

        if problem in str2func.keys():
            with Timer(times, bitwidth):
                result = str2func[problem](number1, number2)
                result = log_expectation(result)
                tf.test.experimental.sync_devices()
        else:
            raise ValueError(f"Unknown problem: {problem}")

    with open(make_path(device, problem) / "times.yaml", "w+") as f:
        yaml.dump(times, f, default_flow_style=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])

    args = parser.parse_args()

    problems = ["sum", "le", "eq"]
    max_bitwidth = 20

    for p in problems:
        if GPUS and args.device == "gpu":
            tf.config.experimental.set_visible_devices(GPUS[0], "GPU")
        else:
            tf.config.experimental.set_visible_devices([], "GPU")
        run_comparison(args.device, p, max_bitwidth)
