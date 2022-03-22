"""Просто кейсы"""

import sys
from copy import deepcopy
from functools import reduce

from buildingBlocks.pickDumps.load_dump import get_data

import buildingBlocks.default.EvolutionEntities as Ee
import buildingBlocks.default.Tokens as Tokens
import buildingBlocks.Globals.GlobalEntities as Ge
import buildingBlocks.Synthesis.Chain as Chain
import buildingBlocks.Synthesis.Synthesizer as Syn

from buildingBlocks.Globals.supplementary.FrequencyProcessor import FrequencyProcessor4TimeSeries as fp

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster

# -6
data = get_data(idx=-2)


# constants = data['constants']
# grid = data['grid']
#
# Ge.set_full_constant(constants)
#
# ind = data['individ']

# ind.structure = ind.structure[:-1]

# for idx, token in enumerate(ind.structure):
#     if isinstance(token, Tokens.Imp):
#         ind.structure[idx] = Tokens.ImpComplex(pattern=ind.structure[idx])
#         ind.structure[idx].init_structure_from_pattern(grid)
#         ind.structure[idx].fitness = ind.structure[idx].pattern.fitness

# print(ind.formula())
# individuals = [ind]
# sys.exit()

# def param_distr(ind, Cimp):
#     clusterer_value = Chain.ClustererPulses(1.2,
#                                             params=dict(grid=grid))
#     clusterer_gaps = Chain.ClustererGaps(distance_threshold=0.8)
#     coder = Chain.Coder2(clusterer_value=clusterer_value, clusterer_gaps=clusterer_gaps,
#                          individ=ind, params=dict(grid=grid))
#     coder.encode()
#
#     for id in range(2):
#         simp_cluster = list(filter(lambda x: x.id_cluster == id, Cimp.structure))
#         # simp_values = list(map(lambda x: x.value(grid), simp_cluster))
#
#         fig = plt.figure('params distr ' + str(id))
#         axs = fig.subplots(3, 2)
#
#         for i in range(6):
#             params = np.array(list(map(lambda x: x.param(idx=i), simp_cluster)))
#             if i == 1:
#                 params = params[1:] - params[:-1]
#             # print(params)
#
#             if i > 1:
#                 params += np.random.normal(0, 0.05*params[0], params.size)
#
#             a = i // 2
#             b = i%2
#             print(a, b)
#             axs[a, b].hist(params, bins=len(params)//2, density=True, label=simp_cluster[0].params_description[i]['name'])
#             axs[a, b].grid(True)
#             axs[a, b].set_xlabel('value')
#             axs[a, b].set_ylabel('hist')
#             axs[a, b].legend()
#         fig.tight_layout()
#
#         fig = plt.figure('ts and pulses ' + str(id))
#         ts = constants['target']
#         plt.plot(grid, ts, color='black', label='original')
#         plt.plot(grid, reduce(lambda x, y: x + y, list(map(lambda x: x.value(grid), simp_cluster))),
#                  color='orange', label='similar tokens in model')
#
#         plt.xlabel('time')
#         plt.ylabel('time series')
#         plt.grid(True)
#         plt.legend()
#
#     plt.show()
#
# param_distr(ind, ind.structure[0])
# sys.exit()



optimizer = data['optimizer']
grid = data['grid']
wmax = 1/(5*(grid[1]-grid[0]))
target = data['target']
target -= target.mean()

Ge.set_constants(target=target)

pareto_level = optimizer.pareto_levels.levels[0]

individuals = list(map(lambda ind: ind.vals, pareto_level))
individuals = list(filter(lambda ind: len(ind.structure) > 1, individuals))

#TODO превращать с ограничением на максимальную частоту и минимальную амплитуду
# for ind in individuals:
#     for idx, token in enumerate(ind.structure):
#         if isinstance(token, Tokens.Imp):
#             ind.structure[idx] = Tokens.ImpComplex(pattern=ind.structure[idx])
#             ind.structure[idx].init_structure_from_pattern(grid)
#             ind.structure[idx].fitness = ind.structure[idx].pattern.fitness

sort_idxs = np.argsort(list(map(lambda ind: len(ind.structure), individuals)))
individuals = [individuals[idx] for idx in sort_idxs[::-1]]
# print(individuals[0].formula())

experiment = {}

# individuals = list(filter(lambda ind: len(ind.structure) == 9, individuals))
# assert individuals

for iiiidx, ind in enumerate(individuals[:1]):
    print(ind.formula())
    experiment[iiiidx] = {}
    experiment[iiiidx]['individ'] = deepcopy(ind)

    experiment[iiiidx]["threshold_value"] = 2
    experiment[iiiidx]["threshold_gaps"] = 0.5

    target_token = list(filter(lambda x: x.mandatory != 0, ind.structure))[0]
    tokens = list(filter(lambda x: x.mandatory == 0, ind.structure))

    clusterer_value = Chain.ClustererPulses(distance_threshold=experiment[iiiidx]["threshold_value"],
                                            params=dict(grid=grid))
    clusterer_gaps = Chain.ClustererGaps(distance_threshold=experiment[iiiidx]["threshold_gaps"])
    coder = Chain.Coder2(clusterer_value=clusterer_value, clusterer_gaps=clusterer_gaps,
                         individ=ind, params=dict(grid=grid))
    mc = Chain.BayesianChain()

    instructions_high = {
                    'Sin': dict(Amplitude=lambda param, grid: param + 0*np.random.normal(0, 0.01*param, grid.shape),
                                # Phase=lambda param, grid: param + 0.1*param*np.sin(0.1*grid),
                                Frequency=lambda param, grid: param - 0.1*param/(0.1*param*(grid+0.0001))
                                                              * np.cos(0.1*param*grid)),
                    # 'Power': dict(Amplitude=lambda param, grid: np.random.normal(param, 0.0*param, grid.shape))
                    # 'Imp': dict(Amplitude=lambda param, grid: param + np.random.normal(0, 0.01*param, grid.shape))
                }
    residuals = ind.value(grid)
    residuals -= residuals.mean()

    figr = plt.figure('residuals')
    axs = figr.subplots(2, 1)
    axs[0].plot(residuals)
    axs[1].hist(residuals)

    syc = Syn.Synthesizer(individ=ind, grid=grid, coder=coder, markov_chain=mc,
                          instructions_high=instructions_high, residuals=residuals)
    syc.fit()

    experiment[iiiidx]['instructions_high'] = str(instructions_high)

    # только когда остатки адекватные


    ts1 = -target_token.value(grid)#-target_token.value(grid)
    ts1 -= ts1.mean()
    ts2 = reduce(lambda x, y: x+y, list(map(lambda x: x.value(grid), tokens)))
    ts2 -= ts2.mean()

    f = lambda x: np.var(np.abs(x))

    w1, s1 = fp.fft(grid, ts1)
    q1 = f(s1 - s1)/f(s1)
    tq1 = np.var(ts1-ts1)/np.var(ts1)

    experiment[iiiidx]['target'] = dict(ts=ts1, spec=np.abs(s1), w=w1, quality_spec=q1, quality_time=tq1)

    w2, s2 = fp.fft(grid, ts2)
    q2 = f(np.abs(s1) - np.abs(s2))/f(np.abs(s1))
    tq2 = np.var(ts1 - ts2) / np.var(ts1)

    experiment[iiiidx]['model'] = dict(ts=ts2, spec=np.abs(s2), w=w2, quality_spec=q2, quality_time=tq2)


    synts = {}
    for i in range(1):
        # tmp_residuals = deepcopy(residuals)
        # np.random.shuffle(tmp_residuals)

        dt = grid[1] - grid[0]
        new_grid = np.arange(0, 3 * grid.max(), dt)

        ts3 = syc.predict(new_grid)
        ts3 = ts3[:len(grid)]
        ts3 -= ts3.mean()
        w3, s3 = fp.fft(grid, ts3)
        tq3 = np.var(ts1 - ts3) / np.var(ts1)
        q3 = f(np.abs(s1) - np.abs(s3)) / f(np.abs(s1))

        # s3 = np.abs(s3)
        synts[i] = dict(ts=ts3, spec=np.abs(s3), w=w3, quality_spec=q3, quality_time=tq3)

    experiment[iiiidx]['synthetic'] = synts


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
#             r'C:\Users\marko\Desktop\делишки\pyscripts\mergeEstarTP\FEDOT.Algs\buildingBlocks\pickDumps\synt_experiments.pkl',
#             mode) as file:
#         data = {
#             'experiment': experiment,
#             'grid': grid,
#             'target': target
#         }
#         pickle.dump(data, file)
# sys.exit()












sts = np.array([synts[key]['ts'] for key in synts.keys()])
sspec = np.array([synts[key]['spec'] for key in synts.keys()])
sqs = np.array([synts[key]['quality_spec'] for key in synts.keys()])


fig = plt.figure('orig and synthetic')
# plt.title('orig and synthetic')
# fig.set_tight_layout(True)
axs = fig.subplots(3, 1, sharex=True, sharey=True)


ts = [experiment[iiiidx]['target']['ts'], experiment[iiiidx]['model']['ts']]
# ax = [None for _ in range(3)]
labels = ['original', 'model', 'synthetic']
colors = ['blue', 'orange', 'green']

for i in range(3):
    if i == 2:
        # axs[i].plot(grid, sts.min(axis=0))
        # axs[i].plot(grid, sts.max(axis=0))
        axs[i].fill_between(grid, sts.min(axis=0), sts.max(axis=0), alpha=0.75, label='bounds')
        axs[i].plot(grid, sts.mean(axis=0), color='red', label='synthetic mean', linewidth=0.5)
    else:
        axs[i].plot(grid, ts[i], label=labels[i], color=colors[i], linewidth=0.5)
    axs[i].grid(True)
    axs[i].set_xlabel('time')
    axs[i].set_ylabel('amplitude')
    axs[i].legend()

fig.align_labels(axs)


fig_sp = plt.figure('spectra')
axs = fig_sp.subplots(3, 1, sharex=True, sharey=True)

w = experiment[iiiidx]['target']['w']
s = [experiment[iiiidx]['target']['spec'], experiment[iiiidx]['model']['spec']]
q = [experiment[iiiidx]['target']['quality_spec'], experiment[iiiidx]['model']['quality_spec']]

for i in range(3):
    if i == 2:
        axs[i].plot(w, sspec.mean(axis=0), color='red', label='synthetic mean: quality {}'.format(sqs.mean()), linewidth=0.5)
        axs[i].fill_between(w, sspec.min(axis=0), sspec.max(axis=0), alpha=0.75, label='bounds')
    else:
        axs[i].plot(w, np.abs(s[i]), label=labels[i]+': quality='+str(q[i]), color=colors[i], linewidth=0.5)
    axs[i].grid(True)
    axs[i].set_xlabel('frequency')
    axs[i].set_ylabel('amplitude')
    axs[i].legend()
fig_sp.align_labels(axs)


fig_distr = plt.figure('distributions')
axs = fig_distr.subplots(3, 1, sharex=True, sharey=True)

for i in range(3):
    if i == 2:
        axs[i].hist(sts.min(axis=0), bins=len(sts.min(axis=0))//50, alpha=0.5, color='green', label='min')
        axs[i].hist(sts.mean(axis=0), bins=len(sts.min(axis=0)) // 50, alpha=0.5, color='red', label='mean')
        axs[i].hist(sts.max(axis=0), bins=len(sts.min(axis=0)) // 50, alpha=0.5, color='blue', label='max')
    else:
        axs[i].hist(ts[i], bins=len(ts[i])//50, label=labels[i], color=colors[i])
    axs[i].grid(True)
    axs[i].set_xlabel('value')
    axs[i].set_ylabel('hist')
    axs[i].legend()
fig_distr.align_labels(axs)

plt.show()
#
#
#
#
#
# dt = grid[1] - grid[0]
# new_grid = np.arange(0, 20*grid.max(), dt)
# synt = syc.predict(new_grid)[:int(0.8*len(new_grid))]
# synt -= synt.mean()
#
# plt.figure('long synt')
# plt.plot(synt, linewidth=0.5)
#
# plt.figure('long spec')
# s = np.abs(np.fft.fft(synt))
# plt.plot(s)

# plt.show()
