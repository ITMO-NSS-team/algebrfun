import os
import sys
from itertools import groupby
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns

root_dir = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(root_dir)

import numpy as np

from cafe.tokens.tokens import Constant
from cafe.tokens.tokens import Sin
from cafe.tokens.tokens import Power
from cafe.tokens.tokens import ComplexToken
from cafe.tokens.tokens import Term

from cafe.evolution.entities import Equation
from cafe.evolution.entities import PopulationOfEquations

from cafe.operators.builder import create_operator_map

# dx = Term(data=np.load("examples//temperature//test//dudx.npy").reshape(-1), name='du/dx')
# dt = Term(data=np.load("examples//temperature//test//dudt.npy").reshape(-1), name='du/dt', mandatory=True)
# dx2 = Term(data=np.load("examples//temperature//test//d2udx2.npy").reshape(-1), name='d2u/dx2')
# terms = [dt, dx, dx2]

x1 = np.load("examples//expressions//test2//x1.npy")
x2 = np.load("examples//expressions//test2//x2.npy")
temp = np.array(list(product(x1, x2)))
grid = np.array([temp[:, 0], temp[:, 1]])

expr = Term(data=np.ones((grid[0].shape[-1])), name="expres")
target = Term(data=-1 * np.load("examples//expressions//test2//target.npy").reshape(-1), name='target', mandatory=True)
terms = [expr, target]

# plt.plot(grid, u.data)
# plt.show()

# plt.plot(grid, du.data)
# plt.show()

token1 = Constant()
token2 = Sin()
token3 = Power()

build_settings = {
    'mutation': {
        'simple': dict(intensive=2, increase_prob=1),
    },
    'crossover': {
        'simple': dict(intensive=1, increase_prob=0.3)
    },
    'tokens': [token1, token3],
    'population': {
        'size': 10
    },
    'terms': terms,
    'lasso':{
        'regularisation_coef': 10**(-6)
    },
    'optimizer':{
        "eps": 0.1
    },
    'shape': (100, 100),
    'target': target,
    'log_file': "examples\\logeq.txt"
}


individ = Equation(max_tokens=10)
# tkn1 = token3.copy()
# tkn1.params = np.array([[2.5, 0.0], [0.0, 1.0]])
# tkn2 = token3.copy()
# tkn2.params = np.array([[-1.5, 1.0], [0.0, 0.0]])
# tkn3 = ComplexToken()
# tkn3.tokens = [Power(params=np.array([[0.4, 1.0], [0.0, 0.0]])), Power(params=np.array([[1.0, 0.0], [0.0, 1.0]]))]
# tkn4 = token3.copy()
# tkn4.params = np.array([[-0.4, 1.0], [0.0, 1.0]])
# tkn5 = token1.copy()
# tkn5.params = np.array([[-0.4], [0.0]])
# tkns = [tkn1, tkn2, tkn3, tkn4, tkn5]

# for tkn in tkns:
#     t = expr.copy()
#     t.expression_token = tkn
#     individ.structure.append(t)

# individ.structure.append(target)

# create_operator_map(grid, individ, build_settings)

# print(individ.formula())
# individ.apply_operator("VarFitnessIndivid")
# print(individ.fitness)

# exit(0)

create_operator_map(grid, individ, build_settings)

population = PopulationOfEquations(iterations=50)

population.evolutionary()
cur_ind = None

for ind in population.structure:
    print(ind.formula(), ind.fitness)
    if cur_ind is None or cur_ind.fitness > ind.fitness:
        cur_ind = ind


expressions = dict((k, list(i)) for k, i in groupby(cur_ind.structure, key=lambda elem: elem.name_))

print(expressions)

plt.plot(population.anal)
plt.show()

for key in expressions.keys():
    print(key)
    value = []
    for elem in expressions[key]:
        print(elem.expression_token.name())
        it_val = elem.expression_token.value(np.array(grid))
        print(it_val.shape)
        # print(it_val)
        if grid.shape[-1] != len(it_val):
            it_val = it_val * np.ones(shape=(grid.shape[-1]))
            it_val = np.array(it_val, dtype=float)
        if len(value) == 0:
            value = it_val
            continue
        value += it_val
    print(np.linalg.norm(value - target.data))
    plt.title(key)
    # try:
    f, axs = plt.subplots(1, 2)
    sns.heatmap(value.reshape(build_settings['shape']), ax=axs[0])
    sns.heatmap(-1 * target.data.reshape(build_settings['shape']), ax=axs[1])
    # except Exception as e:
        # print(str(e))
        # plt.plot(grid, value, label="Received data")
    # plt.show()
    plt.legend()
    plt.savefig(f"{key}.png")
print(cur_ind.fitness)