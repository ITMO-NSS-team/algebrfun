"""Интерфейс к загрузке данных"""


from buildingBlocks.default.EvolutionEntities import *
from buildingBlocks.default.Tokens import *
import moea_dd.forMoeadd.entities.EvolutionaryEntities as Ee
import moea_dd.forMoeadd.entities.Objectives as Objs
from moea_dd.src.moeadd import *
from moea_dd.src.moeadd_supplementary import *

import pickle


load_data = []

# path = r'C:\Users\marko\Desktop\делишки\pyscripts\mergeEstarTP\FEDOT.Algs\moea_dd\forMoeadd\pickDumps\dumpss.pkl'
path = r'C:\Users\marko\Desktop\делишки\pyscripts\mergeEstarTP\FEDOT.Algs\buildingBlocks\pickDumps\dumpss.pkl'
# path = r'C:\Users\marko\Desktop\делишки\pyscripts\mergeEstarTP\FEDOT.Algs\buildingBlocks\pickDumps\synt_experiments.pkl'

with open(path, 'rb') as file:
    while True:
        try:
            load_data.append(pickle.load(file))
        except:
            break


def get_data(idx: int) -> dict:
    try:
        return load_data[idx]
    except IndexError:
        raise IndexError('There is no data with this index')

# data = get_data(-3)
#
# grid = data['grid'],
# target = data['constants']['target']
#
# import matplotlib.pyplot as plt
# plt.plot(target)
# plt.show()
#
#
# decision = input('пиклим? [n/]y')
# if decision == 'y':
#     import pickle
#     decision = input('перезаписываем? [n/]y')
#     if decision == 'y':
#         mode = 'wb'
#     else:
#         mode = 'ab'
#     with open(
#             r'C:\Users\marko\Desktop\делишки\pyscripts\mergeEstarTP\FEDOT.Algs\examples\data\ts_samples.pkl',
#             mode) as file:
#         ndata = {
#             'grid': data['grid'],
#             'target': data['constants']['target']
#         }
#         pickle.dump(ndata, file)