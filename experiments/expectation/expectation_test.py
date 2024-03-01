import os
import gc
import time
import tensorflow as tf
import pickle

from plia import construct_pint, log_expectation


device = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = device


class Timer:
    def __init__(self, times):
        self.times = times

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args):
        t = time.time() - self.start_time
        self.times.append(t)
        print("%s: %.4fs" % ("time", t))


str2func = {
    "addition": lambda x, y: x + y,
    "le": lambda x, y: x < y,
    "eq": lambda x, y: x == y,
}


def run_comparison(name, max_bitwidth=24):
    """Start with a dry run to not time TF backend initialisation cost"""
    bitwidth = 1
    number1 = construct_pint(tf.random.uniform((2**bitwidth,), minval=0, maxval=1), 0)
    number2 = construct_pint(tf.random.uniform((2**bitwidth,), minval=0, maxval=1), 0)
    number1 + number2

    times = []
    for bitwidth in range(max_bitwidth):
        # maybe this helps with memory?
        gc.collect()

        """
        Watch out in this experiment! Tensorflow compiles and caches some operations in the background!
        For example, since the addition is used in the comparisons, it will use the cached operation from the first run!

        Run all of these both separately and when compiled?

        Additional info: bitwidth of 24 uses +- 15GB of VRAM and seems to be max for GPU
        """

        number1 = construct_pint(tf.random.uniform((2**bitwidth,)), 0)
        number2 = construct_pint(tf.random.uniform((2**bitwidth,)), 0)

        if name in str2func.keys():
            with Timer(times):
                result = str2func[name](number1, number2)
                result = log_expectation(result)
        else:
            raise ValueError(f"Unknown name: {name}")

    with open(
        f"experiments/comparison/results/comparison_test_{name}_device_{device}.pkl",
        "wb",
    ) as f:
        pickle.dump(times, f)


if __name__ == "__main__":
    # run_comparison("le")
    run_comparison("addition")
    # run_comparison("eq")
