import os
import numpy as np
import tensorflow as tf

from plia.pint import Categorical


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


ID_LENGTH = 5

probs = np.ones([10, 10]) * 0.1
probs[:, 0] = 0.99
probs[:, 1:] = 0.01 / 9


id = [np.]