"""Просто кейсы"""

from buildingBlocks.pickDumps.load_dump import get_data

import buildingBlocks.default.EvolutionEntities as Ee
import buildingBlocks.default.Tokens as Tokens
import buildingBlocks.Globals.GlobalEntities as Ge
import buildingBlocks.Synthesis.Chain as Chain

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster


data = get_data(idx=-1)


constants = data['constants']
grid = data['grid']

Ge.set_full_constant(constants)

ind = data['individ']

for token in ind.structure:
    print(token.name())
    plt.figure('end')
    if token.name_ == 'target':
        plt.plot(-token.value(grid), label=token.name_)
    else:
        plt.plot(token.value(grid), label=token.name_)
plt.figure('end')
plt.plot(ind.value(grid), label='sum')
# plt.plot(token1.value(grid)*token1.value(grid))
plt.legend()

for idx, token in enumerate(ind.structure):
    if isinstance(token, Tokens.ImpComplex):
        plt.figure('check{}'.format(idx))
        plt.plot(token.value(grid) + 2, label=token.name_)
        for subtoken in token.structure:
            plt.plot(subtoken.value(grid))
        plt.legend()
        plt.grid(True)




target_token = list(filter(lambda x: x.mandatory != 0, ind.structure))[0]
complex_tokens = list(filter(lambda x: isinstance(x, Tokens.ImpComplex), ind.structure))

clusterer_value = Chain.ClustererPulses(min_samples=1, eps=0.1, params=dict(grid=grid))
clusterer_gaps = cluster.DBSCAN(min_samples=1, eps=0.1)
coder = Chain.Coder(clusterer_value=clusterer_value, clusterer_gaps=clusterer_gaps,
                    individ=ind, params=dict(grid=grid))
labels = coder.encode()
print('labels: ', labels)

mc = Chain.MarkovChain()
mc.fit(labels)

checked_states = []
info = {}
for key, value in mc.transitions.items():
    state_count = {}
    for state in value:
        if state not in checked_states:
            state_count[state] = value.count(state)
    info[key] = state_count

for key, value in info.items():
    print('{}: {}'.format(key, value))


new_labels = mc.generate(super_state=labels[0], n_samples=len(labels))

new_imps = coder.decode(new_labels)

new_ind = type(ind)(structure=new_imps)
for imp in new_imps:
    plt.figure('new_imps')
    plt.plot(imp.value(grid))


for token_idx, token in enumerate(ind.structure):
    print(token.name())

    # if token.name_ == 'target':
    #     plt.plot(-token.value(grid), label=token.name_)
    # else:
    #     plt.plot(token.value(grid), label=token.name_)
    if isinstance(token, Tokens.ImpComplex):
        imps = token.structure
        colors = ('red', 'blue', 'black', 'green', 'orange', 'y')
        for idx, imp in enumerate(imps):
            plt.figure(token.name_ + ' -imps ' + str(token_idx))
            plt.plot(grid, imp.value(grid), color=colors[imp.id_cluster % len(colors)])


# ind.del_substructure(target_token)
cind = type(ind)(structure=complex_tokens)
plt.figure('values')
plt.plot(cind.value(grid))
plt.plot(new_ind.value(grid))



plt.show()

