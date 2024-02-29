import os
import time
import pickle

from plia import construct_pint, log_expectation


device = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = device


def run_comparison(name, max_bitwidth=24):
    """Start with a dry run to not time TF backend initialisation cost"""
    bitwidth = 1
    number1 = construct_pint([1 / 2**bitwidth - 1 for _ in range(2**bitwidth)], 0)
    number2 = construct_pint([1 / 2**bitwidth - 1 for _ in range(2**bitwidth)], 0)

    addition = number1 + number2

    times = []
    for bitwidth in range(max_bitwidth):
        """
        Watch out in this experiment! Tensorflow compiles and caches some operations in the background!
        For example, since the addition is used in the comparisons, it will use the cached operation from the first run!

        Run all of these both separately and when compiled?

        Additional info: bitwidth of 24 uses +- 15GB of VRAM and seems to be max for GPU
        """

        """
        Note, most of the time for GPU is actually spent on initialising the two numbers.
        Do we perhaps pull stuff from CPU to GPU?
        """

        if name == "addition":
            addition_time = time.time()
            number1 = construct_pint(
                [1 / 2**bitwidth - 1 for _ in range(2**bitwidth)], 0
            )
            number2 = construct_pint(
                [1 / 2**bitwidth - 1 for _ in range(2**bitwidth)], 0
            )
            addition = number1 + number2
            times.append(time.time() - addition_time)
            print(times[-1])
        elif name == "le":
            le_time = time.time()
            number1 = construct_pint(
                [1 / 2**bitwidth - 1 for _ in range(2**bitwidth)], 0
            )
            number2 = construct_pint(
                [1 / 2**bitwidth - 1 for _ in range(2**bitwidth)], 0
            )
            le = number1 <= number2
            times.append(time.time() - le_time)
            print(times[-1])
        elif name == "exp":
            exp_time = time.time()
            number1 = construct_pint(
                [1 / 2**bitwidth - 1 for _ in range(2**bitwidth)], 0
            )
            number2 = construct_pint(
                [1 / 2**bitwidth - 1 for _ in range(2**bitwidth)], 0
            )
            exp = log_expectation(number1 + number2)
            times.append(time.time() - exp_time)
            print(times[-1])
        elif name == "eq":
            eq_time = time.time()
            number1 = construct_pint(
                [1 / 2**bitwidth - 1 for _ in range(2**bitwidth)], 0
            )
            number2 = construct_pint(
                [1 / 2**bitwidth - 1 for _ in range(2**bitwidth)], 0
            )
            eq = number1 == number2
            times.append(time.time() - eq_time)
            print(times[-1])
        else:
            raise ValueError(f"Unknown name: {name}")

    with open(
        f"experiments/comparison/results/comparison_test_{name}_device_{device}.pkl",
        "wb",
    ) as f:
        pickle.dump(times, f)


if __name__ == "__main__":
    # run_comparison("le")
    run_comparison("exp")
    # run_comparison("eq")
