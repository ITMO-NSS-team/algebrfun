"""Кластеризатор одиночных пульсов по форме и по расположению в последовательности.
Кодировщики/декодировщики импульсов в соответствии с их кластерами.
Простейшие реализации Марковских цепей для закодированных состояний"""

from copy import deepcopy

import buildingBlocks.default.Tokens as Tokens
import buildingBlocks.Globals.GlobalEntities as Ge

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import random


class ClustererPulses:
    def __init__(self, distance_threshold=1, params: dict = None):
        self.distance_threshold = distance_threshold
        self.params = params

        self.points_maxs = []

    def _get_points(self, tokens):
        grid = self.params['grid']
        # durations = list(map(lambda token: token.param(name='Pulse front duration') +
        #                      token.param(name='Pulse recession duration'), tokens))
        # max_duration = min(np.max(durations), grid.max() - grid.min())
        # points = np.array(list(map(lambda token: token.value(grid))))
        pulse_starts = list(map(lambda token: token.param('Pulse start'), tokens))
        for idx, token in enumerate(tokens):
            token.set_param(grid[1], name='Pulse start')
        points = list(map(lambda token: token.value(grid), tokens))
        for idx, token in enumerate(tokens):
            token.set_param(pulse_starts[idx], name='Pulse start')

        points = list(map(lambda point: point[point != 0], points))
        idx_max_duration = np.argmax(list(map(lambda point: len(point), points)))
        max_duration = len(points[idx_max_duration])
        for idx, point in enumerate(points):
            point_duration = len(point)
            if idx != idx_max_duration and point_duration < max_duration:
                points[idx] = np.append(point, np.zeros(max_duration-point_duration))
        points = np.array(points)
        self.points_maxs.append(points.max())
        points /= np.max(self.points_maxs)
        # points /= len(points[0])
        return points

    def fit(self, tokens: list):
        points = self._get_points(tokens)
        # model = cluster.DBSCAN(min_samples=self.min_samples, eps=self.eps)
        model = cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=self.distance_threshold,
                                                compute_full_tree=True)
        model.fit(points)
        # for idx, token in enumerate(tokens):
        #     token.cluster_label = model.labels_[idx]
        print('cluster labels: ', model.labels_.max())

        # TODO для кластеров из одного токена сделать фит предикт в один из существующих кластеров
        # Визуализация результатов (вместо логов)
        # fig = plt.figure('points' + str(np.random.uniform()))
        # cmap = plt.get_cmap('gnuplot')
        # n = model.labels_.max() + 1
        # colors = [cmap(i) for i in np.linspace(0, 1, n)]
        # labels = []
        # axs = fig.subplots(1, 2)
        # for idx, point in enumerate(points):
        #     if model.labels_[idx] not in labels:
        #         axs[0].plot(point, color=colors[model.labels_[idx]], label='cluster ' + str(model.labels_[idx]))
        #         axs[1].plot(tokens[idx].value(self.params['grid']), color=colors[model.labels_[idx]], label='cluster ' + str(model.labels_[idx]))
        #         labels.append(model.labels_[idx])
        #     else:
        #         axs[0].plot(point, color=colors[model.labels_[idx]])
        #         axs[1].plot(tokens[idx].value(self.params['grid']), color=colors[model.labels_[idx]])
        #     # plt.legend(loc=1)
        #     for ax in axs:
        #         ax.set_title('form clustering')
        #         ax.set_xlabel('time index')
        #         ax.set_ylabel('amplitude')
        #         ax.grid(True)
        #         # ax.legend(loc=1)
        #     plt.tight_layout()

        return model.labels_


class ClustererGaps(ClustererPulses):
    def __init__(self, distance_threshold=0.2):
        super().__init__(distance_threshold=distance_threshold)

    def _get_points(self, tokens):
        pulse_starts = list(map(lambda imp: imp.param(name='Pulse start'), tokens))
        points = np.array(list(map(lambda pair: pair[1]-pair[0], zip(pulse_starts[:-1], pulse_starts[1:]))))
        self.gaps_data = dict(gaps=points)
        # if len(points) != 0:
        points_for_fit = points/points.max()
        # else:
        #     points_for_fit = points
        points_for_fit = points_for_fit.reshape(len(points), 1)
        # points_for_fit /= len(points_for_fit[0])
        return points_for_fit

    def fit(self, tokens: list):
        points = self._get_points(tokens)
        # model = cluster.DBSCAN(min_samples=self.min_samples, eps=self.eps)
        # if len(points) <= 1:
        #     return np.zeros(len(points))
        model = cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=self.distance_threshold,
                                                compute_full_tree=True)
        model.fit(points)
        return model.labels_


class Coder:
    def __init__(self, individ, clusterer_value, clusterer_gaps, params: dict = None): # todo поменять инд на лист сложных токенов
        self.individ = individ
        self.clusterer_value = clusterer_value
        self.clusterer_gaps = clusterer_gaps
        self.params = params
        _, self.complex_imps = self._get_complex_imps()
        self.all_imps = self._get_all_imps()
        self._prepare_imps()

        self.decode_data = {}

    def _get_complex_imps(self):
        idxs_complex_imps = list(filter(lambda idx: isinstance(self.individ.structure[idx], Tokens.ImpComplex),
                                        range(len(self.individ.structure))))
        complex_imps = list(filter(lambda token: isinstance(token, Tokens.ImpComplex),
                                   self.individ.structure))

        # todo complex_imps must be sorted by fitness
        fits = list(map(lambda token: token.fitness, complex_imps))
        complex_imps = [complex_imps[i] for i in np.argsort(fits)]
        fits = [fits[i] for i in np.argsort(fits)]
        for i in range(len(fits)-1):
            assert fits[i] <= fits[i+1], 'not sorted'

        return idxs_complex_imps, complex_imps

    def _get_all_imps(self):
        all_imps = []
        for complex_imp in self.complex_imps:
            all_imps.extend(complex_imp.structure)
        # sorted(all_imps, key=lambda x: x.param(name='Pulse start'))
        pulse_starts = list(map(lambda imp: imp.param(name='Pulse start'), all_imps))
        idxs = np.argsort(pulse_starts)
        all_imps = [all_imps[idx] for idx in idxs]
        return all_imps

    def _prepare_imps(self):
        # grid = self.params['grid']
        # for idx, imp in enumerate(self.all_imps):
        #     val = imp.value(grid)
        #     non_zero_grid = grid[val != 0]
        #     imp.set_param(non_zero_grid.min(), name='Pulse start')
        #     imp.set_param(non_zero_grid.max() -
        #                   (non_zero_grid.min() +
        #                    imp.param(name='Pulse front duration')), name='Pulse recession duration')
        #     sorted(self.all_imps, key=lambda x: x.param(name='Pulse start'))
        for complex_imp in self.complex_imps:
            for imp in complex_imp.structure:
                imp.set_param(imp.param(name='Amplitude')*complex_imp.param(name='Amplitude'),
                              name='Amplitude')

    def _label_gaps(self):
        # pulse_starts = list(map(lambda imp: imp.param(name='Pulse start'), self.all_imps))
        pulse_starts = list(map(lambda imp: imp.param(name='Pulse start'), self.all_imps))
        gaps = np.array(list(map(lambda pair: pair[1]-pair[0], zip(pulse_starts[:-1], pulse_starts[1:]))))

        gaps_for_fit = gaps/gaps.max()
        gaps_for_fit = gaps_for_fit.reshape(len(gaps), 1)

        clusterer = self.clusterer_gaps
        clusterer.fit(gaps_for_fit)
        labels = clusterer.labels_
        # print('time labels: ', labels)

        for idx, imp in enumerate(self.all_imps):
            try:
                imp.id_gap = labels[idx-1]
            except IndexError:
                # imp.id_gap = clusterer.fit_predict(np.array([gaps.mean()]))[0] # todo надо бы сделать среднее время из его кластера
                imp.id_gap = clusterer.fit_predict(np.array([imp.param(name='Pulse start')]))[0]
        self.gaps_data = dict(gaps=gaps, gaps_labels=labels)

        # colors = ('red', 'blue', 'black', 'green', 'orange', 'y')
        # for idx, imp in enumerate(self.all_imps):
        #     plt.figure('time gaps')
        #     plt.plot(imp.value(self.params['grid']) + imp.id_gap, color=colors[imp.id_gap % len(colors)])

        # sum_gaps = 0
        # for idx, imp in enumerate(all_imps):
        #     try:
        #         imp.id_gap = all_imps[idx+1].param(name='Pulse start') - all_imps[idx].param(name='Pulse start')
        #         sum_gaps += imp.id_gap
        #     except IndexError:
        #         imp.id_gap = sum_gaps/idx

    def _label_complex_imps(self):
        for idx, complex_imp in enumerate(self.complex_imps):
            complex_imp.id_ImpComplex = idx
            for idx_imp, imp in enumerate(complex_imp.structure):
                imp.id_ImpComplex = complex_imp.id_ImpComplex

    def _label_values(self):
        for idx, complex_imp in enumerate(self.complex_imps):
            cluster_labels = self.clusterer_value.fit(complex_imp.structure)
            for idx_imp, imp in enumerate(complex_imp.structure):
                imp.id_cluster = cluster_labels[idx_imp]

    def _label_tokens(self):
        self._label_gaps()

        info = []
        for idx, complex_imp in enumerate(self.complex_imps):
            cluster_labels = self.clusterer_value.fit(complex_imp.structure)
            complex_imp.id_ImpComplex = idx
            info.append(cluster_labels.max())
            for idx_imp, imp in enumerate(complex_imp.structure):
                imp.id_cluster = cluster_labels[idx_imp]
                imp.id_ImpComplex = idx

        # colors = ('red', 'blue', 'black', 'green', 'orange', 'y', 'purple', 'brown')
        # colors = [
        #     'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
        #     'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
        # cmap = plt.get_cmap('gnuplot')
        # colors = [cmap(i) for i in np.linspace(0, 1, np.array(info).max()+1)]
        # lines = ('-', '--', '-o')
        # names = []
        # for idx, imp in enumerate(self.all_imps):
        #     plt.figure('time gaps')
        #     plt.title('clustering')
        #     name = str(imp.id_cluster)
        #     if name not in names:
        #         plt.plot(imp.value(self.params['grid']) + imp.id_gap, color=colors[imp.id_cluster],#color=colors[imp.id_cluster % len(colors)],
        #                  label=name)
        #         names.append(name)
        #     else:
        #         plt.plot(imp.value(self.params['grid']) + imp.id_gap, color=colors[imp.id_cluster % len(colors)])
        #     plt.xlabel('time')
        #     plt.ylabel('amplitude/gap cluster')
        #     # plt.legend()
        #     plt.grid(True)

    def encode(self):
        self._label_tokens()
        labels = []
        for imp in self.all_imps:
                labels.append(tuple((imp.id_ImpComplex, imp.id_cluster, imp.id_gap)))
        return labels

    def decode(self, labels, grid=None, init_pulse_start=0):
        if grid is None:
            grid = self.params['grid']
        grid_max = grid.max()
        new_imps = []
        for label in labels:
            id_ImpComplex, id_cluster, id_gap = label
            complex_imp = list(filter(lambda cimp: cimp.id_ImpComplex == id_ImpComplex, self.complex_imps))[0]
            imps = list(filter(lambda imp: imp.id_cluster == id_cluster, complex_imp.structure))
            new_imp = np.random.choice(imps).copy()
            gap = np.random.choice(self.gaps_data['gaps'][self.gaps_data['gaps_labels'] == id_gap])

            try:
                new_imp.set_param(new_imps[-1].param(name='Pulse start') + gap, name='Pulse start')
            except IndexError:
                new_imp.set_param(np.random.uniform(init_pulse_start, init_pulse_start+gap), name='Pulse start')

            # break decoding extra labels

            new_imps.append(new_imp)
            if new_imp.param(name='Pulse start') >= grid_max:
                break
        return new_imps
# todo мы генерим цепью маркова батч семплов, потом смотрим где старт последнего пульса, если недостаточно
# то генерим еще, если сгенерили лишка то нужно поставить проверку в декодере что старт пульса превышает грид макс


class MarkovChain:
    def __init__(self, transitions: dict = None):
        if transitions is None:
            transitions = {}
        self.transitions = transitions

    def fit(self, states: list):
        for idx, state in enumerate(states):
            if state not in self.transitions.keys():
                self.transitions[state] = []
            try:
                self.transitions[state].append(states[idx + 1])
            except IndexError:
                pass

    def generate(self, super_state=None, n_samples=1):
        all_states = list(self.transitions.keys())
        if super_state is None:
            # super_state = random.choice(all_states)
            super_state = all_states[0]
        generated = [super_state]
        for _ in range(n_samples):
            concurrent_state = generated[-1]
            states = self.transitions[concurrent_state]
            if states:
                new_state = random.choice(states)
            else:
                cluster_states = list(filter(lambda x: (x[0] == concurrent_state[0] and
                                                        x[1] == concurrent_state[1] and
                                                        x != concurrent_state),
                                             all_states))
                if cluster_states:
                    state = random.choice(cluster_states)
                    new_state = random.choice([state])
                else:
                    cluster_states = list(filter(lambda x: (x[0] == concurrent_state[0] and
                                                            x != concurrent_state),
                                                 all_states))
                    if cluster_states:
                        state = random.choice(cluster_states)
                        new_state = random.choice([state])
                    else:
                    # raise
                        new_state = random.choice(all_states)
            # print('{} -> {}'.format(generated[-1], new_state))
            generated.append(new_state)
        # print('generated: ', generated)
        return generated, generated[-1]


class Coder2(Coder):

    def _label_values(self):
        cluster_labels = self.clusterer_value.fit(self.all_imps)
        for idx_imp, imp in enumerate(self.all_imps):
            imp.id_cluster = cluster_labels[idx_imp]

    def _label_gaps(self):
        n_clusters = max(list(map(lambda imp: imp.id_cluster, self.all_imps)))
        for idx in range(n_clusters + 1):
            imp_cluster = list(filter(lambda imp: imp.id_cluster == idx, self.all_imps))
            if len(imp_cluster) == 1:
                imp_cluster[0].id_gap = 0
                self.decode_data[idx] = {}
                self.decode_data[idx]['imp_cluster'] = imp_cluster
                self.decode_data[idx]['gaps'] = np.array([imp_cluster[0].param('Pulse start')])
                self.decode_data[idx]['gaps_labels'] = np.zeros(1)
                continue
            self.decode_data[idx] = {}
            self.decode_data[idx]['imp_cluster'] = imp_cluster
            cluster_labels = self.clusterer_gaps.fit(imp_cluster)
            for idx_imp, imp in enumerate(imp_cluster):
                try:
                    imp.id_gap = cluster_labels[idx_imp-1]
                except IndexError:
                    imp.id_gap = self.clusterer_gaps.fit_predict(np.array([imp.param(name='Pulse start')]))[0]

            print('time labels: ', cluster_labels.max())

            self.decode_data[idx]['gaps'] = deepcopy(self.clusterer_gaps.gaps_data['gaps'])
            self.decode_data[idx]['gaps_labels'] = deepcopy(cluster_labels)

            # complex_imp.gaps_data = deepcopy(self.clusterer_gaps.gaps_data)
            # complex_imp.gaps_data['gaps_labels'] = deepcopy(cluster_labels)


            # plt.figure('gaps' + str(idx))
            # grid = self.params['grid']
            # cmap = plt.get_cmap('gnuplot')
            # n = cluster_labels.max()+1
            # colors = [cmap(i) for i in np.linspace(0, 1, n)]
            # labels = []
            # for idx_imp, imp in enumerate(imp_cluster):
            # # for idx_imp, imp in enumerate(complex_imp.structure):
            #     try:
            #         if imp.id_gap not in labels:
            #             plt.plot(imp.value(grid), color=colors[imp.id_gap], label='cluster ' + str(imp.id_gap))
            #             labels.append(imp.id_gap)
            #         else:
            #             plt.plot(imp.value(grid), color=colors[imp.id_gap])
            #         # plt.legend()
            #         plt.title('gaps clustering')
            #         plt.xlabel('time index')
            #         plt.ylabel('amplitude')
            #
            #     except IndexError:
            #         pass

    def _label_tokens(self):
        self._label_complex_imps()
        self._label_values()
        self._label_gaps()

    def encode(self):
        self._label_tokens()
        labels = []
        for imp in self.all_imps:
                labels.append(tuple((imp.id_cluster, imp.id_gap)))
        return labels

    def decode(self, labels, grid=None, init_pulse_start=0, generated_imps=None):
        if grid is None:
            grid = self.params['grid']
        if generated_imps is None:
            generated_imps = []
        grid_max = grid.max()
        for label in labels:
            if label is None:
                continue
            # id_ImpComplex, id_cluster, id_gap = label
            id_cluster, id_gap = label
            # complex_imp = list(filter(lambda cimp: cimp.id_ImpComplex == id_ImpComplex, self.complex_imps))[0]
            # imps = list(filter(lambda imp: imp.id_cluster == id_cluster, complex_imp.structure))
            imps = self.decode_data[id_cluster]['imp_cluster']
            new_imp = random.choice(imps).copy()
            # gap = np.random.choice(complex_imp.gaps_data['gaps'][complex_imp.gaps_data['gaps_labels'] == id_gap])
            gap = np.random.choice(self.decode_data[id_cluster]['gaps'][self.decode_data[id_cluster]['gaps_labels'] == id_gap])

            # for imp_idx in range(len(generated_imps)-1, -1, -1):
            #     if new_imp.id_ImpComplex == generated_imps[imp_idx].id_ImpComplex:
            #         new_imp.set_param(generated_imps[imp_idx].param(name='Pulse start') + gap, name='Pulse start')
            #         break
            # else:
            #     #todo тут скрыт какой то великий баг
            #
            #     # new_imp.set_param(init_pulse_start + new_imp.param('Pulse start'), name='Pulse start')
            #     # new_imp.set_param(np.random.uniform(init_pulse_start, init_pulse_start + gap), name='Pulse start')
            #     # new_imp.set_param(init_pulse_start + gap, name='Pulse start')
            #
            #     new_imp.set_param(complex_imp.structure[0].param('Pulse start'), name='Pulse start')

            for imp_idx in range(len(generated_imps)-1, -1, -1):
                if new_imp.id_cluster == generated_imps[imp_idx].id_cluster:
                    new_imp.set_param(generated_imps[imp_idx].param(name='Pulse start') + gap, name='Pulse start')
                    break
            else:
                new_imp.set_param(imps[0].param('Pulse start'), name='Pulse start')

            # break decoding extra labels

            generated_imps.append(new_imp)
            if new_imp.param(name='Pulse start') >= grid_max:
                break
        # return generated_imps


class BayesianChain:
    def __init__(self, transitions: dict = None):
        if transitions is None:
            transitions = {}
        self.transitions = transitions

    def fit(self, states: list):
        # print('fitting..')
        self.super_state_len = max(list(map(lambda x: x[0], states)))+1
        super_state = tuple([None for _ in range(self.super_state_len)])

        for idx, state in enumerate(states):
            if super_state not in self.transitions.keys():
                self.transitions[super_state] = []
            try:
                self.transitions[super_state].append(state)
                next_super_state = list(super_state)
                next_super_state[state[0]] = state
                # for tmp_idx in range(state[0]+1, self.super_state_len):
                #     next_super_state[tmp_idx] = None
                next_super_state = tuple(next_super_state)
                super_state = next_super_state
            except IndexError:
                pass
        self.all_super_states = list(self.transitions.keys())
        self.all_states = list(set([s for item in self.all_super_states for s in item]))
        # for key, value in self.transitions.items():
        #     print('{}: {}'.format(key, value))

    def generate(self, super_state=None, init_state=None, n_samples=1):
        # cluster_maxs = []
        # for i in range(len(all_states[0])):
        #     labels = list(map(lambda x: x[i], all_states))
        #     cluster_maxs.append(max(labels))
        # vals = []
        # probs = []
        # for i in range(len(cluster_maxs)):
        #     vals.append(list(range(cluster_maxs[i])))

        if super_state is None:
            super_state = self.all_super_states[0]
            generated = list(super_state)
        else:
            generated = []
        for _ in range(n_samples):
            concurrent_state = super_state
            # for i in range(self.super_state_len-1, -2, -1):
            #     try:
            #         states = self.transitions[concurrent_state]
            #         break
            #     except KeyError:
            #         concurrent_state = list(concurrent_state)
            #         concurrent_state[i] = None
            #         concurrent_state = tuple(concurrent_state)
            try:
                new_state = new_state
            except:
                new_state = init_state
            while True:
                try:
                    states = self.transitions[concurrent_state]
                    new_state = random.choice(states)
                    break
                except KeyError or IndexError or ValueError:
                    concurrent_state = list(concurrent_state)
                    # зависящий от длины состояния пульса код
                    if np.random.uniform() <= 0.9:
                        # try:
                        #     tmp = list(filter(lambda s: s is not None and s[0] == new_state[0] and s[1] == new_state[1],
                        #                       self.all_states))
                        #     concurrent_state[new_state[0]] = random.choice(tmp)
                        # except IndexError or ValueError:
                        tmp = list(filter(lambda s: s is not None and s[0] == new_state[0], self.all_states))
                        concurrent_state[new_state[0]] = random.choice(tmp)
                    else:
                        concurrent_state = random.choice(self.all_super_states)
                    concurrent_state = tuple(concurrent_state)

            # print('{} -> {}'.format(generated[-1], new_state))
            generated.append(new_state)
            next_super_state = list(super_state)
            next_super_state[new_state[0]] = new_state
            # for tmp_idx in range(new_state[0] + 1, self.super_state_len):
            #     next_super_state[tmp_idx] = None
            next_super_state = tuple(next_super_state)
            super_state = next_super_state

        # print('generated: ', generated)
        # print('super_state: ', super_state)
        return generated, super_state
