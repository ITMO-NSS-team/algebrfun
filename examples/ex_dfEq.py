import os
import random
import sys

root_dir = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(root_dir)

from copy import deepcopy
from functools import reduce
from itertools import product
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from load_data import get_data

import buildingBlocks.Builder.OperatorsBuilder as Ob
import buildingBlocks.Globals.GlobalEntities as Bg
import moea_dd.forMoeadd.entities.EvolutionaryEntities as Ee
import moea_dd.forMoeadd.entities.Objectives as Objs
from buildingBlocks.baseline.BasicEvolutionaryEntities import Term
from buildingBlocks.default.EvolutionEntities import (DEquation, Equation,
                                                      Subpopulation)
from buildingBlocks.default.Tokens import (Constant, Imp, ImpComplex, Power,
                                           Product, Sin)
from buildingBlocks.Globals.GlobalEntities import (get_full_constant,
                                                   set_constants)
from buildingBlocks.Globals.supplementary.FrequencyProcessor import \
    FrequencyProcessor4TimeSeries as fp
from buildingBlocks.Synthesis import Chain
from buildingBlocks.Synthesis.Synthesizer import Synthesizer
from moea_dd.src.moeadd import *
from moea_dd.src.moeadd_supplementary import *

## Set tokens from which algorithm will be built model-expression
# Constant token is the target that will be approximated by other tokens
# ImpComplex is a set of splitted single pulses obtained from periodic impulse

token1 = Constant(val=None, name_='target', mandatory=1)
token2 = Sin(optimize_id=1, name_='Sin')
token3 = Imp(optimize_id=1, name_='Imp')
token4 = Power(optimize_id=2, name_='Power')

pattern = Imp(optimize_id=1)
impComplex_token = ImpComplex(pattern=pattern, optimize_id=3)


## Choose dataset
# There are 3 datasets 
# of series with different structure. 
# Good meta parameters (build_settings) of the algorithm are selected for each of them.

# data = get_data(0)
build_settings = {
    'mutation': {
        'simple': dict(intensive=1, increase_prob=1),
        'complex': dict(prob=0., threshold=0.1, complex_token=impComplex_token)
    },
    'crossover': {
        'simple': dict(intensive=1, increase_prob=0.3)
    },
    'tokens': [token1, token2, token4],
    'population': {
        'size': 10
    }
}

### Time series without seasonality

i = 2 #3
data = get_data(i)
build_settings = {
    'mutation': {
        'simple': dict(intensive=1, increase_prob=1),
        'complex': dict(prob=0.5, threshold=0.5, complex_token=impComplex_token)
    },
    'crossover': {
        'simple': dict(intensive=1, increase_prob=0.3)
    },
    'tokens': [token1, token2, token3],
    'population': {
        'size': 10
    },
    'lasso':{
        'regularisation_coef': 10**(-6)
    }
}

## Get target and grid on which target will be approximated

# random.seed(10)
# grid = np.array([data['grid']])
# target = data['target']
# target -= target.mean()

'''
# begin - проверка функций, которые мы знаем
x = np.linspace(0, 2*np.pi, 100)
y = x / 2
xy = np.array(list(product(x, y)))
XX, YY = np.meshgrid(x, y)
target = np.array([np.sin(el[0] + el[1]) for el in xy])
# target = np.array([np.sin(el[0] + el[1]) for el in xy])

# target.reshape(-1)

grid = np.array([xy[:, 0], xy[:, 1]])
target -= target.mean()
# ---- end

# begin - ЛЕД
ice_file = pd.read_csv("examples//ice_data//input_data.csv", header=None)
mask_file = np.loadtxt("examples//ice_data//bathymetry.npy")
ice_data = ice_file.iloc[395:450, 160:215]
mask_data = np.bool_(mask_file[395:450, 160:215])
print("ice data", ice_data)

x = np.arange(0, 55)
y = np.arange(0, 55)
xy = np.array(list(product(x, y)))
target = ice_data.to_numpy().reshape(-1)
grid = np.array([xy[:, 0], xy[:, 1]])
target -= target.mean()
# ------- end

'''
# begin pde
grid = np.load("examples//pde//t.npy")
u = Term(0, np.load("examples//pde//u.npy"), 'u')
du = Term(1, np.load("examples//pde//du.npy").reshape(-1), 'du/dt')
noize_one = Term(2, np.random.uniform(-1, 0, 960).reshape(-1), 'noise_one')
noize_two = Term(3, np.random.uniform(-1, 1, 960).reshape(-1), 'noise_two')
# ----- end


shp = (1,960)
set_constants(target=u, shape_grid=shp, pul_mtrx=[du, noize_one])

individ = Equation(max_tokens=10)
Ob.set_operators(np.array([grid]), individ, build_settings)


population = Subpopulation(iterations=10)

# population = PopulationOfDEquations(iterations=10)

population.evolutionary()

print("mi")

inds = population.structure
# idxsort = np.argsort(list(map(lambda x: x.fitness, inds)))
# inds = [inds[i] for i in idxsort]

print("RESULTING")
n = 0
ind = deepcopy(inds[-1].structure[n])
for i, ind in enumerate(inds[-1].structure):
    print(i, ind.formula(), ind.fitness)

# print(ind.formula(), ind.fitness)

residuals = ind.value(grid)


f, axs = plt.subplots(1, 2)
target_draw = u.reshape(shp)
# target_draw[~mask_data] = np.nan
model_draw = residuals.reshape(shp)
# model_draw[~mask_data] = np.nan

pc_test = axs[0].imshow(target_draw)
axs[0].set_title("Input data")
axs[1].imshow(model_draw)
axs[1].set_title("Model")
 
plt.savefig('dftest.png')