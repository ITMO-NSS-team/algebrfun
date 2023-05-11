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
from cafe.tokens.tokens import Cos
from cafe.tokens.tokens import Term

from cafe.evolution.entities import Equation
from cafe.evolution.entities import PopulationOfEquations

from cafe.operators.builder import create_operator_map

def main(grid, terms, target_data, shape_grid):
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
        'log_file': "examples\\logeq_caf.txt"
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
    plt.savefig(f"examples//pde//anal.png")
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
        np.save(f"examples//pde//{name_file}_res.npy", value.reshape(-1))
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
    out_file = open("examples//pde//result.txt", 'w')
    out_file.write(f"{cur_ind.formula()}, {cur_ind.fitness}")


if __name__ == "__main__":
    # загрузка данных времени и производных из файлов

    grid = np.load("examples//pde//t.npy")
    u = Term(data=np.load("examples//pde//u.npy"), name='u')
    du = Term(data=np.load("examples//pde//du.npy").reshape(-1), name='du/dt')
    const_matr = Term(data=-1*np.ones(960), name='constante', mandatory=True)
    grid = np.array([grid])
    terms = [u, du, const_matr]

    # данные для температуры (200, 30)
    # grid_t = np.load("examples//temperature//convection_t.npy")[:200]
    # grid_x = np.load("examples//temperature//convection_x.npy")
    # tx = np.array(list(product(grid_t, grid_x)))
    # grid = np.array([tx[:, 0], tx[:, 1]])

    # dx = Term(data=np.load("examples//temperature//dudx.npy").reshape(-1), name='du/dx')
    # dt = Term(data=np.load("examples//temperature//dudt.npy").reshape(-1), name='du/dt', mandatory=True)
    # dx2 = Term(data=np.load("examples//temperature//d2udx2.npy").reshape(-1), name='d2u/dx2')
    # terms = [dt, dx, dx2]

    # (101, 50)
    # grid_t = np.load("examples//temperature//test//t.npy")
    # grid_x = np.load("examples//temperature//test//x.npy")
    # tx = np.array(list(product(grid_t, grid_x)))
    # grid = np.array([tx[:, 0], tx[:, 1]])

    # dx = Term(data=np.load("examples//temperature//test//dudx.npy").reshape(-1), name='du/dx')
    # dt = Term(data=np.load("examples//temperature//test//dudt.npy").reshape(-1), name='du/dt', mandatory=True)
    # dx2 = Term(data=np.load("examples//temperature//test//d2udx2.npy").reshape(-1), name='d2u/dx2')
    # terms = [dt, dx, dx2]

    try:
        population = PopulationOfEquations(iterations=10)
        main(grid, terms, const_matr, (960,))
    except KeyboardInterrupt:
        plt.plot(population.anal)
        plt.savefig(f"examples//pde//anal.png")

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
            np.save(f"examples//pde//{name_file}_res.npy", value.reshape(-1))

        print(cur_ind.fitness)
        out_file = open("examples//pde//result.txt", 'w')
        out_file.write(f"{cur_ind.formula()}, {cur_ind.fitness}")