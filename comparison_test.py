import os
import time
import pickle

from categorical import Categorical


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def run_comparison(name, max_bitwidth=24):
    for bitwidth in range(max_bitwidth, max_bitwidth + 1):
        addition_times = []
        le_times = []
        exp_times = []

        """ 
        Watch out in this experiment! Tensorflow compiles and caches some operations in the background! 
        For example, since the addition is used in the comparisons, it will use the cached operation from the first run!

        Run all of these both separately and when compiled?

        Additional info: bitwidth of 24 uses +- 15GB of VRAM and seems to be max for GPU
        """

        number1 = Categorical([1 / 2 ** bitwidth - 1 for _ in range(2 ** bitwidth)], 0, 2 ** bitwidth - 1)
        number2 = Categorical([1 / 2 ** bitwidth - 1 for _ in range(2 ** bitwidth)], 0, 2 ** bitwidth - 1)

        if name == 'addition':
            addition_time = time.time()
            addition = number1 + number2
            addition_times.append(time.time() - addition_time)
            print(addition_times[-1])
        elif name == 'le':
            le_time = time.time()
            le = number1 <= number2
            le_times.append(time.time() - le_time)
            print(le_times[-1])
        elif name == 'exp':
            exp_time = time.time()
            exp = number1.E
            exp_times.append(time.time() - exp_time)
            print(exp_times[-1])


    # with open('comparison_test.pkl', 'wb') as f:
    #     pickle.dump({'addition_times': addition_times, 'le_times': le_times, 'exp_times': exp_times}, f)

run_comparison("addition")