import numpy as np
import os
import sys

root_dir = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(root_dir)

import matplotlib.pyplot as plt
from cafe.tokens.tokens import Sin
from cafe.tokens.tokens import Cos
from cafe.tokens.tokens import Power
from cafe.tokens.tokens import Constant

grid = np.load("examples//params//t.npy")
# target = np.sin(grid)
target = grid ** 2

token = Power()

ampls = token._find_initial_approximation_(np.array([grid]), target, 10)

print(ampls)
print(token.variable_params)