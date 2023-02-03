import os
import sys

root_dir = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(root_dir)

import numpy as np

from cafe.tokens.tokens import Constant
from cafe.tokens.tokens import Sin
from cafe.tokens.tokens import Power
from cafe.tokens.tokens import Term

from cafe.evolution.entities import Equation
from cafe.evolution.entities import PopulationOfEquations

from cafe.operators.builder import create_operator_map


# загрузка данных времени и производных из файлов

grid = np.load("examples//pde//t.npy")
u = Term(data=np.load("examples//pde//u.npy"), name='u')
du = Term(data=np.load("examples//pde//du.npy").reshape(-1), name='du/dt')
const_matr = Term(data=np.ones(960), name='constante', mandatory=True)

token1 = Constant()
token2 = Sin()
token3 = Power()

build_settings = {
    'mutation': {
        'simple': dict(intensive=1, increase_prob=1),
    },
    'crossover': {
        'simple': dict(intensive=1, increase_prob=0.3)
    },
    'tokens': [token1, token2, token3],
    'population': {
        'size': 10
    },
    'terms': [u, du, const_matr],
    'lasso':{
        'regularisation_coef': 10**(-6)
    },
    'shape': (1, 960) 
}


individ = Equation(max_tokens=10)
create_operator_map(np.array([grid]), individ, build_settings)

population = PopulationOfEquations(iterations=15)

population.evolutionary()


