import os
import time
import pickle

from plia.pint import PInt


device = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = device


def run_comparison(name, max_bitwidth=23):
    times = []
    for bitwidth in range(max_bitwidth + 1):
        """
        Watch out in this experiment! Tensorflow compiles and caches some operations in the background!
        For example, since the addition is used in the comparisons, it will use the cached operation from the first run!

        Run all of these both separately and when compiled?

        Additional info: bitwidth of 24 uses +- 15GB of VRAM and seems to be max for GPU
        """

        number1 = PInt(
            [1 / 2**bitwidth - 1 for _ in range(2**bitwidth)], 0, 2**bitwidth - 1
        )
        number2 = PInt(
            [1 / 2**bitwidth - 1 for _ in range(2**bitwidth)], 0, 2**bitwidth - 1
        )

        if name == "addition":
            addition_time = time.time()
            addition = number1 + number2
            times.append(time.time() - addition_time)
            print(times[-1])
        elif name == "le":
            le_time = time.time()
            le = number1 <= number2
            times.append(time.time() - le_time)
            print(times[-1])
        elif name == "exp":
            exp_time = time.time()
            exp = (number1 + number2).E
            times.append(time.time() - exp_time)
            print(times[-1])
        elif name == "eq":
            eq_time = time.time()
            eq = number1 == number2
            times.append(time.time() - eq_time)
            print(times[-1])
        else:
            raise ValueError(f"Unknown name: {name}")

    with open(f"comparison_test_{name}_device_{device}.pkl", "wb") as f:
        pickle.dump(times, f)


run_comparison("eq")
