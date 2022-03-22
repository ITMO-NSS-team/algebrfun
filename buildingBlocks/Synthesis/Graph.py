"""Просто кейсы"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation

import buildingBlocks.Globals.GlobalEntities as Ge

from buildingBlocks.pickDumps.load_dump import get_data

data = get_data(idx=-1)

grid = data['grid']
target = data['target']

Ge.set_constants(target=target)

experiment = data['experiment']

inds_len = [len(experiment[key]['individ'].structure) for key in experiment.keys()]

sorted_ind_idxs = np.argsort(inds_len)

nn = -1
ex = experiment[sorted_ind_idxs[nn]]

ind = ex['individ']
print(ind.formula())

# plt.figure('formula')
#
# for token in ind.structure:
#     print(token.name())
#     if token.name_ == 'target':
#         plt.plot(-token.value(grid), label=token.name_)
#     else:
#         print(token.fitness)
#         plt.plot(token.value(grid), label=token.name_)



individs = [experiment[key]['individ'] for key in experiment.keys()]
fitnesses = np.array([indd.fitness for indd in individs])
lengths = np.array([len(indd.structure) for indd in individs])
qualities = np.array([np.array([experiment[key]['synthetic'][key1]['quality_spec']
                               for key1 in experiment[key]['synthetic'].keys()])
                      for key in experiment.keys()])
t_qualities = np.array([np.array([experiment[key]['synthetic'][key1]['quality_time']
                                 for key1 in experiment[key]['synthetic'].keys()])
                       for key in experiment.keys()])

q = qualities/t_qualities
# q = q.mean(axis=1)

# -3
# q[lengths == 3] = 0.465
# q[lengths == 9] = 0.37

sorted_idxs = np.argsort(lengths)



# plt.figure('pareto models')
# plt.plot(lengths[sorted_idxs], fitnesses[sorted_idxs], '-o', label='data model')
#
# # plt.figure('pareto synthetics')
# plt.plot(lengths[sorted_idxs], q[sorted_idxs], '-o', label='synthetic data')
#
# plt.xlabel('model size')
# plt.ylabel('quality model/synthetic metric')
# plt.title('Pareto frontier')
# plt.grid(True)
# plt.legend()






synts = ex['synthetic']


wn = 0
wN = len(ex['target']['w'])#//2 + 100

sts = np.array([synts[key]['ts'] for key in synts.keys()])
sspec = np.array([synts[key]['spec'][wn:wN] for key in synts.keys()])
# sqs = np.array([synts[key]['quality_spec'] for key in synts.keys()])

font_prop = font_manager.FontProperties(size=8)
# name = 'elec'
name = 'temp'

fig = plt.figure('orig and synthetic')
# plt.title('orig and synthetic')
# fig.set_tight_layout(True)
axs = fig.subplots(3, 1, sharex=True, sharey=True)


ts = [ex['target']['ts'], ex['model']['ts']]
# ax = [None for _ in range(3)]
labels = ['original', 'model value', 'synthetic']
colors = ['blue', 'orange', 'green']

for i in range(2):
    axs[i].plot(grid, ts[i], label=labels[i], color=colors[i], linewidth=0.5)
    axs[i].grid(True)
    axs[i].set_xlabel('time')
    axs[i].set_ylabel('amplitude')
    axs[i].legend(loc=1, prop=font_prop)

i = 2
a1, = axs[i].plot([], [], color='red', linewidth=0.5)
a2 = axs[i].fill_between(grid, sts.min(axis=0), sts.max(axis=0), alpha=1, label='min-max bounds')
a1.set_data(grid, sts[2])
a1.set_label('synthetic sample')
a2.set_label('min-max bounds')



axs[i].grid(True)
axs[i].set_xlabel('time')
axs[i].set_ylabel('amplitude')
axs[i].legend(loc=1, prop=font_prop)


fig.align_labels(axs)



def init():
    a1.set_data([], [])
    return a1,
def animate(i):
    a1.set_data(grid, sts[i])
    a1.set_label('synthetic sample: quality={}'.format(round(q[sorted_ind_idxs[nn]][i], 3)))
    axs[2].legend(loc=1, prop=font_prop)
    return a1,

anim = FuncAnimation(fig, animate, init_func=init, frames=20, interval=1500, blit=True)

anim.save(r'C:\Users\marko\Desktop\ИТМО диплом\src\images\{}.gif'.format(name))

fig_sp = plt.figure('spectra')
print(type(fig_sp))
axs = fig_sp.subplots(3, 1, sharex=True, sharey=True)
print(type(axs))

w = ex['target']['w'][wn:wN]
s = [ex['target']['spec'][wn:wN], ex['model']['spec'][wn:wN]]
qq = [ex['target']['quality_spec'], ex['model']['quality_spec']]

labels = ['original spectrum', 'model value spectrum', 'synthetic spectrum']

for i in range(2):
    axs[i].plot(w, np.abs(s[i]), label=labels[i]+': quality='+str(round(qq[i], 3)), color=colors[i], linewidth=0.5)
    axs[i].grid(True)
    axs[i].set_xlabel('frequency')
    axs[i].set_ylabel('amplitude')
    axs[i].legend(loc=1, prop=font_prop)
fig_sp.align_labels(axs)

i = 2
a1, = axs[i].plot([], [], color='red', linewidth=0.5)
a2 = axs[i].fill_between(w, sspec.min(axis=0), sspec.max(axis=0), alpha=0.75)

# a1.set_label('synthetic sample: quality={}'.format(round(q[sorted_ind_idxs[nn]][2], 3)))
a2.set_label('min-max bounds')

axs[i].grid(True)
axs[i].set_xlabel('frequency')
axs[i].set_ylabel('amplitude')
axs[i].legend(loc=1, prop=font_prop)

def init1():
    a1.set_data([], [])
    return a1,
def animate1(i):
    a1.set_data(w, sspec[i])
    a1.set_label('synthetic sample spectrum: quality={}'.format(round(q[sorted_ind_idxs[nn]][i], 3)))
    axs[2].legend(loc=1, prop=font_prop)
    return a1,

anim = FuncAnimation(fig_sp, animate1, init_func=init1, frames=20, interval=1500, blit=True)
anim.save(r'C:\Users\marko\Desktop\ИТМО диплом\src\images\{}_spectra.gif'.format(name))

# fig_distr = plt.figure('distributions')
# axs = fig_distr.subplots(3, 1, sharex=True, sharey=True)
#
# for i in range(3):
#     if i == 2:
#             axs[i].hist(sts.min(axis=0), bins=len(sts.min(axis=0))//50//2, density=True, alpha=0.5, color='green', label='synthetic min')
#             axs[i].hist(sts.mean(axis=0), bins=len(sts.min(axis=0)) // 50//2, density=True, alpha=0.5, color='red', label='synthetic mean')
#             axs[i].hist(sts.max(axis=0), bins=len(sts.min(axis=0)) // 50//2, density=True, alpha=0.5, color='blue', label='synthetic max')
#     else:
#         axs[i].hist(ts[i], bins=len(ts[i])//25//2, density=True, label=labels[i], color=colors[i])
#     axs[i].grid(True)
#     axs[i].set_xlabel('value')
#     axs[i].set_ylabel('hist')
#     axs[i].legend()
# fig_distr.align_labels(axs)
#
# plt.show()