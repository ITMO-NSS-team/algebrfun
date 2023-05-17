import numpy as np
import os
import matplotlib.pyplot as plt
import sys

root_dir = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(root_dir)

import matplotlib.pyplot as plt
from cafe.tokens.tokens import Sin
from cafe.tokens.tokens import Cos
from cafe.tokens.tokens import Power
from cafe.tokens.tokens import Constant

grid = np.load("examples//params//t.npy")
# grid = np.arange(1, 5)
# target = np.sin(grid)
target = np.power(grid, 2)
# target = np.cos(grid)

token = Power()

ampls = token._find_initial_approximation_(np.array([grid]), target, 4)

ampls = np.array(ampls)

x = sorted(ampls, reverse=True)
y = token.variable_params.reshape(-1)

print(x)
print(y)

plt.scatter(y, x)
plt.show()

plt.plot(target, label="target")
print(2*np.pi*y[0] )
# plt.plot(np.cos(2*np.pi*y[0] * grid), label=f"find freq.={np.round(2*np.pi*y[0], 2)}")
plt.plot(grid**y[0])
plt.legend()
plt.show()