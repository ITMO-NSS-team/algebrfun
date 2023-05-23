import numpy as np
import pandas as pd
from itertools import product
import os
import sys
from itertools import groupby
import matplotlib.pyplot as plt

root_dir = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(root_dir)

from cafe.tokens.tokens import Constant
from cafe.tokens.tokens import Sin
from cafe.tokens.tokens import Power
from cafe.tokens.tokens import Cos
from cafe.tokens.tokens import Term

from cafe.evolution.entities import Equation
from cafe.evolution.entities import PopulationOfEquations

from cafe.operators.builder import create_operator_map

def main(grid, terms, target_data, shape_grid):
    dir_path = "examples/kdv/"
    # plt.plot(grid, u.data)
    # plt.show()

    # plt.plot(grid, du.data)
    # plt.show()

    token1 = Constant()
    token2 = Sin()
    token3 = Power()
    token4 = Cos()

    build_settings = {
        'mutation': {
            'simple': dict(intensive=2, increase_prob=1),
        },
        'crossover': {
            'simple': dict(intensive=1, increase_prob=0.3)
        },
        'tokens': [token1, token2, token3, token4],
        'population': {
            'size': 10
        },
        'terms': terms,
        'lasso':{
            'regularisation_coef': 10**(-6)
        },
        'optimizer':{
            "eps": 0.05
        },
        'shape': shape_grid,
        'target': target_data,
        'log_file': f"{dir_path}logeq.txt"
    }


    individ = Equation(max_tokens=10)
    create_operator_map(grid, individ, build_settings)

    # population = PopulationOfEquations(iterations=40)

    population.evolutionary()
    cur_ind = None

    for ind in population.structure:
        print(ind.formula(), ind.fitness)
        if cur_ind is None or cur_ind.fitness > ind.fitness:
            cur_ind = ind


    expressions = dict((k, list(i)) for k, i in groupby(cur_ind.structure, key=lambda elem: elem.name_))

    print(expressions)

    plt.plot(population.anal)
    plt.savefig(f"{dir_path}anal.png")
    # plt.show()

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
        print(f"109: {value.shape}")
        name_file = "_".join(key.split("/"))
        np.save(f"{dir_path}{name_file}_res.npy", value.reshape(-1))
        # plt.title(key)
        # try:
        # sns.heatmap(value.reshape(build_settings['shape']))
        # except Exception as e:
            # print(str(e))
            # plt.plot(grid, value, label="Received data")
        # plt.show()
        # name_file = "_".join(key.split("/"))
        # plt.savefig(f"{name_file}.png")
    print(cur_ind.fitness)
    out_file = open(f"{dir_path}result.txt", 'w')
    out_file.write(f"{cur_ind.formula()}, {cur_ind.fitness}")

if __name__ == "__main__":
    dir_path = "examples/kdv/"
    df = pd.read_csv(f'{dir_path}KdV_sln.csv', header=None)
    dddx = pd.read_csv(f'{dir_path}ddd_x.csv', header=None)
    ddx = pd.read_csv(f'{dir_path}dd_x.csv', header=None)
    dx = pd.read_csv(f'{dir_path}d_x.csv', header=None)
    dt = pd.read_csv(f'{dir_path}d_t.csv', header=None)

    u = df.values
    u = np.transpose(u)

    ddd_x = dddx.values
    ddd_x = np.transpose(ddd_x)
    dd_x = ddx.values
    dd_x = np.transpose(dd_x)
    d_x = dx.values
    d_x = np.transpose(d_x)
    d_t = dt.values
    d_t = np.transpose(d_t)
    
    t = np.linspace(0, 1, u.shape[0])
    x = np.linspace(0, 1, u.shape[1])
    temp = np.array(list(product(t, x)))
    grid = np.array([temp[:, 0], temp[:, 1]])


    # u = Term(data=u.reshape(-1), name="u")
    d_x = u.reshape(-1) * d_x.reshape(-1)
    du_dt = Term(data=-1*d_t.reshape(-1), name="du/dt", mandatory=True)
    du_dx = Term(data=d_x, name="u_du/dx")
    du_dx3 = Term(data=ddd_x.reshape(-1), name="d3u/dx3")
    const_matr = Term(data=np.ones(grid.shape[-1]), name='constante')
    terms = [du_dt, du_dx, du_dx3, const_matr]

    try:
        population = PopulationOfEquations(iterations=10)
        main(grid, terms, du_dt, (101, 101))
    except KeyboardInterrupt:
        plt.plot(population.anal)
        plt.savefig(f"{dir_path}anal.png")

        cur_ind = None

        for ind in population.structure:
            print(ind.formula(), ind.fitness)
            if cur_ind is None or cur_ind.fitness > ind.fitness:
                cur_ind = ind

        expressions = dict((k, list(i)) for k, i in groupby(cur_ind.structure, key=lambda elem: elem.name_))

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
            print(f"109: {value.shape}")
            name_file = "_".join(key.split("/"))
            np.save(f"{dir_path}{name_file}_res.npy", value.reshape(-1))

        print(cur_ind.fitness)
        out_file = open(f"{dir_path}result.txt", 'w')
        out_file.write(f"{cur_ind.formula()}, {cur_ind.fitness}")

