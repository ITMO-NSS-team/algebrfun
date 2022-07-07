from copy import deepcopy, copy
from functools import reduce

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.signal import argrelextrema
from sklearn import cluster
import matplotlib.pyplot as plt

from buildingBlocks.Globals.GlobalEntities import set_constants, get_full_constant
from buildingBlocks.baseline.BasicEvolutionaryEntities import GeneticOperatorIndivid, GeneticOperatorPopulation
from buildingBlocks.default.Tokens import ImpComplex
from buildingBlocks.default.geneticOperators.supplementary.Other import create_tmp_individ, \
    check_or_create_fixator_item, apply_decorator


class ImpComplexTokenParamsOptimizer(GeneticOperatorIndivid):
    """
    Works with an object of the 'ImpComplex' class.
    Optimizes the parameters of all objects of the 'ImpSingle' class for better approximation of input data.
    """

    # def __init__(self, individ, chromo_idx):
    #     self.individ = individ
    #     self.chromo_idx = chromo_idx
    #     self.token = self.individ.chromo[chromo_idx]

    def __init__(self, params=None):
        if params is None:
            params = {}
        add_params = {
            'optimizer': 'DE',
            'optimize_id': None,
            'popsize': 7,
            'eps': 0.005
        }
        for key, value in add_params.items():
            if key not in params.keys():
                params[key] = value
        super().__init__(params=params)
        self._check_params('grid', 'optimizer', 'optimize_id', 'popsize', 'eps')

    @staticmethod
    def _fitness_wrapper(params, *args):
        individ, grid, token = args
        token.params = params
        individ.fitness = None
        individ.apply_operator(name='VarFitnessIndivid')
        return individ.fitness

    def _optimize_complex_token_params(self, individ, complex_token):
        # if complex_token.fixator['self']: возможна перефиксация, но снизу фиксация токенов этого не допускает (коряво)
        #     return
        grid = self.params['grid']
        for token in complex_token.structure:
            if not token.fixator['self']:
                token.fixator['self'] = True
                res = minimize(self._fitness_wrapper, deepcopy(token.params),
                               args=(individ, grid, token), method='Nelder-Mead')
                               # options=dict(maxiter=token._number_params*1000,
                               #              maxfev=token._number_params*1000))
                token.params = res.x
            token.fixator['self'] = True
        complex_token.fixator['self'] = True # убрать отсюда в апплай

    # --------------->exp
    # @staticmethod
    # def fitness_wrapper_all(params, *args):
    #     individ, t, target_T S= args
    #     individ.put_params(params)
    #     return individ.fitness(t, target_TS)
    #
    # def optimize_params_local_all(self, t, target_TS):
    #     params0 = self.token_individ.get_params()
    #     res = minimize(self.fitness_wrapper_all, params0, args=(self.token_individ, t, target_TS), method='Nelder-Mead')
    #     self.token_individ.put_params(res.x)

    # <------------------

    def _get_single_pattern_value(self, complex_token):
        grid = self.params['grid']
        constants = get_full_constant()
        pattern = complex_token.pattern
        T = 1 / pattern.param(name='Frequency')
        T1 = pattern.param(name='Zero part of period') * T
        T2 = T - T1
        fi = pattern.param(name='Phase')
        test_val = pattern.value(grid)
        # test_val = test_val.reshape(constants['shape_grid'])
        mask = np.ones(grid.shape[-1], dtype=bool)
        for i in range(grid.shape[0]):
            cur_mask = np.zeros(grid.shape[-1], dtype=bool)
            cur_mask[(grid[i] >= T[i] - fi[i] * T[i] + T1[i]) & (grid[i] <= T[i] - fi[i] * T[i] + T1[i] + T2[i])] = True
            mask *= cur_mask
        # mask = mask.reshape(constants['shape_grid'])
        single_pattern_value = test_val[mask]
                               # * np.sign(pattern.param(name='Amplitude'))
        try:
            single_pattern_value = single_pattern_value[single_pattern_value > 0]
            assert len(single_pattern_value) != 0, 'hmm'
        except:
            pass
        assert (single_pattern_value != 0).any(), 'hmm'
        return single_pattern_value

    @staticmethod
    def _tokens_fitnesses(individ, tokens):
        fitnesses = []
        for token in tokens:
            individ.add_substructure(token)
            individ.fitness = None
            individ.apply_operator('VarFitnessIndivid')
            fitnesses.append(individ.fitness)
            individ.del_substructure(token)
        return fitnesses

    @staticmethod
    def _get_tmp_individ_with_target_token(individ, complex_token):
        target_tokens = list(filter(lambda token: token.mandatory != 0, individ.structure))
        assert len(target_tokens) == 1, 'There must be only one target token'

        tmp_individ = individ.clean_copy()
        tmp_individ.structure = copy(target_tokens)

        fixed_optimized_tokens_in_structure = list(filter(lambda token: token.fixator['self'] and token.mandatory == 0,
                                                          individ.structure))

        fitnesses = ImpComplexTokenParamsOptimizer._tokens_fitnesses(tmp_individ, fixed_optimized_tokens_in_structure)

        # todo choose best metric for that
        # могут быть баги с синхронизацией параметров паттерна
        complex_token_fitness = ImpComplexTokenParamsOptimizer._tokens_fitnesses(tmp_individ,
                                                                                 [complex_token.pattern])


        idxs_significant_fixed_optimized_tokens_in_structure = list(filter(
            lambda idx: fitnesses[idx] <= complex_token_fitness[0],
            range(len(fitnesses))))
        significant_fixed_optimized_tokens_in_structure = [
            fixed_optimized_tokens_in_structure[idx] for idx in idxs_significant_fixed_optimized_tokens_in_structure
        ]

        tmp_individ.add_substructure(significant_fixed_optimized_tokens_in_structure)
        tmp_individ.add_substructure(complex_token)
        return tmp_individ

    def _find_pulse_starts(self, target, complex_token):
        single_pattern_value = self._get_single_pattern_value(complex_token)
        grid = self.params['grid']

        # опасно если у нас выделился тупо прямоугольный импульс, поэтому добавим дельту
        # single_pattern_value -= single_pattern_value.mean() + 0.001
        # single_pattern_value /= max(single_pattern_value.min(), single_pattern_value.max(), key=abs)
        # target -= target.mean()
        # target /= max(target.min(), target.max(), key=abs)

        # single_pattern_value -= single_pattern_value.min()
        single_pattern_value /= single_pattern_value.max()
        # target -= target.min()
        target /= target.max()

        T = 1 / complex_token.pattern.param(name='Frequency')
        cor = np.correlate(target, single_pattern_value, mode='valid')  # /np.var(pattern)
        cor -= cor.min()
        cor /= max(cor.min(), cor.max(), key=abs)
        # todo дисперсия = двоякая величина. Бывает токенам выгоднее стать плоскими в другом месте, чтоюы дисперсию
        # максимизировать, потому что на том промежутке это действительно выгодней (???). Почему токены в новом месте
        # начинают странно оптимизироваться? Может метрика неверна?
        # cor = target_TS
        # find maximums

        # бывает и такое, что точки оказываются обсолютно одинаковыми, и максимум не находится, но это в идеал.усл.
        # idxs = []
        # for i in range(1, len(cor) - 1):
        #     if cor[i] > cor[i - 1] and cor[i] > cor[i + 1]:# and cor[i] > 0:
        #         idxs.append(i)
        all_idxs = argrelextrema(cor, lambda x1, x2: x1 > x2, mode='wrap')[0]

        # all_idxs = np.array(idxs)

        idxs = list(all_idxs)

        i = 1
        delta = 0.2
        while i < len(idxs):
            if np.all((grid[:, idxs[i]] - grid[:, idxs[i - 1]]) < (1 - delta) * T): # !!!!!!!(?) 
                a = idxs[i] if cor[idxs[i]] < cor[idxs[i - 1]] else idxs[i - 1]
                idxs.remove(a)
            else:
                i += 1

        # all_idxs_train = all_idxs.reshape(len(all_idxs), 1)
        # # todo опасная вещь, "понижает" частоту, ибо экстремумов меньше чем пульсов
        # # todo сделать проверку на то, что переставленные пульсы забирают больше фитнесса чем паттерн
        # n_clusters = min(len(all_idxs), round(grid.max()/T))
        # # if len(all_idxs) < round(grid.max()/T):
        # #     return
        # clusterer = cluster.KMeans(n_clusters=n_clusters)
        # clusterer.fit(all_idxs_train)
        # extremas = []
        # for label in range(clusterer.labels_.max()+1):
        #     extremas.append(all_idxs[clusterer.labels_ == label].max())

        # mas_idxs = np.array(extremas)
        mas_idxs = np.array(idxs)

        # plt.figure('corr' + str(np.random.uniform()))
        # plt.plot(target, label='target')
        # plt.plot(cor, label='correlation')

        # tops = np.zeros(len(cor))
        # tops[all_idxs] = cor[all_idxs]
        # plt.plot(tops, label='alltstarts')

        tops = np.zeros(len(cor))
        tops[mas_idxs] = cor[mas_idxs]
        # plt.plot(tops, label='pulse starts')
        #
        # plt.plot(single_pattern_value, label='pattern')
        # plt.xlabel('time')
        # plt.ylabel('value')
        # plt.grid(True)
        # plt.legend()
        # plt.show()

        ret = grid[:, mas_idxs]

        return ret

    def _init_structure_with_pulse_starts(self, complex_token, pulse_starts):
        grid = self.params['grid']
        complex_token.init_structure_from_pattern(grid)
        init_pulses_starts = np.array(list(map(lambda imp: imp.param(name='Pulse start'), complex_token.structure)))
        used_imps = []

        imp_single_pattern = complex_token.get_substructure(idx=0)
        imp_single_pattern.val = None
        imps = []
        for start in pulse_starts.T:
            # flag = False
            # idxs_candidates = np.argsort(np.abs((init_pulses_starts - start)))
            # # к каждому экстремуму пододвигаем ближайший пульс
            # for idx in idxs_candidates:
            #     if complex_token.structure[idx] not in used_imps:
            #         complex_token.structure[idx].set_param(start, name='Pulse start')
            #         used_imps.append(complex_token.structure[idx])
            #         flag = True
            #         break
            # # если все экстремумумов больльше чем ипульсов, то создаем новые
            # if not flag:
            imps.append(imp_single_pattern.copy())
            # neighbour_pulses_starts = list(map(lambda pulse: pulse.param(name='Pulse start'), idxs_candidates[:2]))
            imps[-1].set_param(start, name='Pulse start')
            # used_imps.append(imps[-1])

        complex_token.structure = imps
        # new_pulses_starts = list(map(lambda imp: imp.param(name='Pulse start'), complex_token.structure))
        # idx_sorted = np.argsort(new_pulses_starts)
        # complex_token.structure = [complex_token.structure[idx] for idx in idx_sorted]
        #
        # # если экстремумов меньше чем пульсов
        # for idx, pulse in enumerate(complex_token.structure):
        #     if pulse not in used_imps:
        #         if idx == 0:
        #             mean_start = complex_token.structure[idx+1].param('Pulse start') - pulse.param('Duration')
        #         elif idx == len(complex_token.structure) - 1:
        #             mean_start = (complex_token.structure[idx-1].param('Pulse start') +
        #                           complex_token.structure[idx-1].param('Duration'))
        #         else:
        #             mean_start = reduce(lambda x, y: x+y,
        #                                 list(map(lambda pulse_idx: complex_token.structure[pulse_idx].param('Pulse start'),
        #                                          [idx-1, idx+1])))/2
        #         pulse.set_param(mean_start, name='Pulse start')

    def _optimize_token_params(self, individ, complex_token):
        grid = self.params['grid']
        extra_tmp_individ = self._get_tmp_individ_with_target_token(individ, complex_token)

        # init_target = -extra_tmp_individ.value(grid)
        # init_target -= init_target.min()

        target = -extra_tmp_individ.value(grid)
        # центрирование и нормализация (fitness - дисперсия, так что центрирование ничего не меняет)
        # target -= target.mean()
        # target /= np.abs(target).max()
        target -= target.min()
        target /= target.max()

        pulse_starts = self._find_pulse_starts(target, complex_token)
        self._init_structure_with_pulse_starts(complex_token, pulse_starts)

        tmp_individ = create_tmp_individ(extra_tmp_individ, [complex_token], target)

        self._optimize_complex_token_params(tmp_individ, complex_token)

        # plt.figure('result {}'.format(np.random.uniform()))
        # plt.plot(target, label='target')
        # # plt.plot(init_target, label='init_target')
        # plt.plot(complex_token.pattern.value(grid), label='pattern')
        # plt.plot(complex_token.value(grid), label='cimp')
        # plt.plot(target - complex_token.value(grid) - 1, label='residuals')
        # for token in complex_token.structure:
        #     plt.plot(token.value(grid)-2)
        # plt.grid(True)
        # plt.legend()
        # plt.show()

    def _choice_tokens_for_optimize(self, individ):
        optimize_id = self.params['optimize_id']
        choiced_tokens = list(filter(lambda token: token.optimize_id == optimize_id and not token.fixator['self'],
                                     individ.structure))
        if len(choiced_tokens) == 0:
            return choiced_tokens

        target_tokens = list(filter(lambda token: token.mandatory != 0, individ.structure))
        assert len(target_tokens) == 1, 'There must be only one target token'

        tmp_individ = individ.clean_copy()
        tmp_individ.structure = copy(target_tokens)
        fitnesses = self._tokens_fitnesses(tmp_individ,
                                           list(map(lambda token: token.pattern, choiced_tokens)))

        sorted_idxs = np.argsort(fitnesses)[::-1]
        choiced_tokens = [choiced_tokens[idx] for idx in sorted_idxs]

        return choiced_tokens

    @apply_decorator
    def apply(self, individ, *args, **kwargs):

        choiced_tokens = self._choice_tokens_for_optimize(individ)
        if len(choiced_tokens) == 0:
            return

        for complex_token in choiced_tokens:
            self._optimize_token_params(individ, complex_token)
        return


class ImpComplexDiscreteTokenParamsOptimizer(ImpComplexTokenParamsOptimizer):
    """
    Works with an object of the 'ImpComplex'class.
    Optimizes the parameters of all objects of the 'ImpSingle' class for better approximation of input data.
    """

    def __init__(self, params=None):
        if params is None:
            params = {}
        add_params = {
            'optimizer': '',
            'optimize_id': None,
            'popsize': 7,
            'eps': 0.005
        }
        for key, value in add_params.items():
            if key not in params.keys():
                params[key] = value
        super().__init__(params=params)
        self._check_params('grid', 'optimizer', 'optimize_id', 'popsize', 'eps')

    @staticmethod
    def _fitness_wrapper(params, *args):
        individ, grid, token = args
        # T0 = token.params[1]
        # try:
        token.params = params.reshape(grid.shape[0], len(params)//grid.shape[0])
        # except:
        #     token.params = params
        # token.params[1] = T0
        # individ.fitness = None
        # individ.apply_operator(name='VarFitnessIndivid', grid=grid)
        # return individ.fitness
        fixed_optimized_tokens_in_structure = list(filter(lambda token: token.fixator['self'],
                                                          individ.structure))
        if len(fixed_optimized_tokens_in_structure) != 0:
            val = reduce(lambda val, x: val + x, list(map(lambda x: x.value(grid),
                                                          fixed_optimized_tokens_in_structure)))
        else:
            val = np.zeros(grid.shape)
        # return np.linalg.norm(val)
        return np.var(val)
        # return np.abs(val - val.mean()).mean()

    def _optimize_complex_token_params(self, individ):
        target_tokens = list(filter(lambda token: token.mandatory != 0, individ.structure))
        tokens_to_optimize = list(filter(lambda token: token.mandatory == 0, individ.structure))
        assert len(target_tokens) == 1, 'There must be only one target token'
        target_token = target_tokens[0]

        pulse_starts = list(map(lambda imp: imp.param(name='Pulse start'), tokens_to_optimize))
        idxs = np.argsort(pulse_starts)
        tokens_to_optimize = [tokens_to_optimize[idx.min()] for idx in idxs]

        # self.random_fig_mark = np.random.uniform()
        # plt.figure('result {}'.format(self.random_fig_mark))
        # for token in tokens_to_optimize:
        #     plt.plot(token.value(self.params['grid']), color='b')
        if len(tokens_to_optimize) == 1:
            tokens_to_optimize[0].fixator['self'] = True
            return

        grid = self.params['grid']
        grid = grid.T
        delta = 0.5
        for token_idx, token in enumerate(tokens_to_optimize):
            tmp_params_description = deepcopy(token.params_description)
            chosen_params = (1, 2)
            for idx in chosen_params:
                # token.set_descriptor(key=idx, descriptor_name='bounds',
                #                      descriptor_value=(token_params[idx]*(1-delta),
                #                                        token_params[idx]*(1+delta)))
                token.set_descriptor(key=idx, descriptor_name='check', descriptor_value=True)

            if token_idx != 0:
                prev_token = tokens_to_optimize[token_idx - 1]
                try:
                    msk = prev_token.value(grid) != 0
                    prev_token_end = [grid[k][msk].max() for k in range(grid.shape[0])]
                except:
                    prev_token_end = prev_token.param(name='Pulse start') + prev_token.param(name='Duration')
            if token_idx != len(tokens_to_optimize) - 1:
                next_token = tokens_to_optimize[token_idx + 1]
                try:
                    msk = next_token.value(grid) != 0
                    next_token_start = [grid[k][msk].min() for k in range(grid.shape[0])]
                except:
                    next_token_start = next_token.param(name='Pulse start')

            token_duration = token.param(name='Duration')
            token_start = token.param(name='Pulse start')
            if token_idx == 0:
                token.set_descriptor(key=1, descriptor_name='bounds',
                                     # descriptor_value=(-token_duration,
                                     #                   next_token_start + token_duration*delta))
                                     descriptor_value=(token_start - token_duration * delta,
                                                       token_start + token_duration * delta))
                token.set_descriptor(key=2, descriptor_name='bounds',
                                     descriptor_value=(np.zeros(len(next_token_start)),
                                                       ((next_token_start
                                                        - token_start) * (1+delta))))
            elif token_idx == len(tokens_to_optimize) - 1:
                token.set_descriptor(key=1, descriptor_name='bounds',
                                     # descriptor_value=(prev_token_end - token_duration*delta,
                                     #                   grid.max()))
                                     descriptor_value=(token_start - token_duration * delta,
                                                       token_start + token_duration * delta))
                token.set_descriptor(key=2, descriptor_name='bounds',
                                     descriptor_value=(np.zeros(len(token_start)),
                                                       (grid.max() + token_duration - token_start)*(1+delta)))
            else:
                token.set_descriptor(key=1, descriptor_name='bounds',
                                     # descriptor_value=(prev_token_end - token_duration*delta,
                                     #                   min(next_token_start + token_duration*delta,
                                     #                       prev_token_end - token_duration*delta + token_duration)))
                                     descriptor_value=(token_start - token_duration * delta,
                                                       token_start + token_duration*delta))
                token.set_descriptor(key=2, descriptor_name='bounds',
                                     descriptor_value=(np.zeros(len(next_token_start)),
                                                       (next_token_start
                                                        - token_start)*(1+delta)))

            bounds = deepcopy(token.get_descriptor_foreach_param(descriptor_name='bounds'))
            try:
                left_grid_bound = np.array(bounds[1])[0, :] + np.array(bounds[2])[0, :]
            except:
                left_grid_bound = bounds[1][0] + bounds[2][0]

            try:
                right_grid_bound = np.array(bounds[1])[1, :] + np.array(bounds[2])[1, :]
            except:
                right_grid_bound = bounds[1][1] + bounds[2][1]

            # grid_optimize = np.array([grid[k] if grid[k] <= right_grid_bound] for k in range(grid.shape[0])])
            msk = np.apply_along_axis(np.any, 1, grid <= right_grid_bound)
            grid_optimize = grid[msk, :]


            constants = get_full_constant()
            new_target_idxs = np.arange(0, grid.shape[0], 1, dtype=int)
            new_target_idxs = new_target_idxs[msk]
            new_target_idxs = new_target_idxs[np.apply_along_axis(np.any, 1, grid_optimize >= left_grid_bound)]
            grid_optimize = grid_optimize[np.apply_along_axis(np.any, 1, grid_optimize >= left_grid_bound), :]
            new_target_value = deepcopy(constants[target_token.name_])[new_target_idxs]

            token.set_param([new_target_value.max()], name='Amplitude')

            target_token = target_tokens[0]
            tmp_target_name = deepcopy(target_token.name_)
            new_target_name = 'ntm'
            set_constants(ntm=new_target_value)
            target_token.name_ = new_target_name

            grid_optimize = grid_optimize.T
            inv_idx = token_idx-1
            while inv_idx >= 0:
                if np.all((tokens_to_optimize[inv_idx].param('Pulse start') +
                        tokens_to_optimize[inv_idx].param('Duration')) < np.array([grid_optimize_iter.min() for grid_optimize_iter in grid_optimize])):
                    tokens_to_optimize[inv_idx].fixator['self'] = False
                    break
                tokens_to_optimize[inv_idx].fixator['val'] = False
                inv_idx -= 1

            
            prms = deepcopy(token.params)
            res = minimize(self._fitness_wrapper, prms.reshape(-1),
                           args=(individ, grid_optimize, token), method='Nelder-Mead')
            # res = differential_evolution(self._fitness_wrapper, deepcopy(bounds),
            #                              args=(individ, grid_optimize, token), popsize=2)

            target_token.name_ = tmp_target_name

            # bounds = deepcopy(token.get_descriptor_foreach_param(descriptor_name='bounds'))
            # try:
            #     res = differential_evolution(self._fitness_wrapper, bounds,
            #                                  args=(individ, grid, token), popsize=3)
            # except ValueError:
            #     pass
            # token.params = res.x
            self._fitness_wrapper(res.x, individ, grid_optimize, token)
            token.params_description = tmp_params_description
            token.fixator['self'] = True

        for token in tokens_to_optimize:
            token.fixator['self'] = True
        individ.apply_operator('LRIndivid1Target')

    def _optimize_token_params(self, individ, complex_token):
        grid = self.params['grid']
        extra_tmp_individ = self._get_tmp_individ_with_target_token(individ, complex_token)

        target = -extra_tmp_individ.value(grid)
        target -= target.min()
        target /= target.max()  

        pulse_starts = self._find_pulse_starts(target, complex_token)
        self._init_structure_with_pulse_starts(complex_token, pulse_starts)

        optimizing_tokens = copy(complex_token.structure)
        tmp_individ = create_tmp_individ(extra_tmp_individ, optimizing_tokens, target)

        self._optimize_complex_token_params(tmp_individ)
        self._remove_small_tokens(complex_token)
        complex_token.fixator['self'] = True


        # plt.figure('result {}'.format(np.random.uniform()))
        # plt.plot(target, label='target')
        # # plt.plot(init_target, label='init_target')
        # plt.plot(0.9*complex_token.pattern.value(grid),
        #          label='impulse before additional optimization')#.format(complex_token.pattern.param(name='Frequency')))
        # plt.plot(complex_token.value(grid), label='impulse after additional optimization')
        # # plt.plot(target - complex_token.value(grid) - 1, label='residuals')
        # # for token in complex_token.structure:
        # #     plt.plot(token.value(grid)-2)
        # plt.grid(True)
        # plt.xlabel('time')
        # plt.ylabel('time series')
        # plt.legend()
        # plt.show()

    @staticmethod
    def _remove_small_tokens(complex_token):
        amps = list(map(lambda token: token.param('Amplitude')[0], complex_token.structure))
        mx_amp = np.max(amps)
        mn_amp = np.min(amps)
        diff = mx_amp - mn_amp
        new_structure = []
        print("part with removed small tokens", mn_amp + 0.3 * diff, mx_amp*0.05)
        for idx, token in enumerate(complex_token.structure):
            if amps[idx] >= 0.06*mx_amp:
                new_structure.append(token)
            # if amps[idx] >= (mn_amp + diff * 0.3):
            #     new_structure.append(token)
        complex_token.structure = new_structure


class ImpSimpleOptimizerIndivid(GeneticOperatorIndivid):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('grid', 'constraints')

    def _set_pulse_start(self, token, target):
        grid = self.params['grid']

        pattern_value = token.value(grid)
        pattern_value = pattern_value[pattern_value > 0]

        if (pattern_value == 0).all():
            return

        pattern_value -= pattern_value.min()
        pattern_value /= pattern_value.max()
        target -= target.min()
        target /= target.max()

        cor = np.correlate(target, pattern_value, mode='valid')  # /np.var(pattern)
        # cor -= cor.mean()
        # cor /= max(cor.min(), cor.max(), key=abs)

        pulse_start = grid[np.argmax(cor)]
        token.set_param(pulse_start, name='Pulse start')

    @staticmethod
    def _fitness_wrapper(params, *args):
        individ, grid, token, constraints = args

        # вешаем ограничения
        for key, value in constraints.items():
            bounds = value
            assert bounds[0] <= bounds[1], 'MIN > MAX'
            params[key] = min(params[key], bounds[1])
            params[key] = max(params[key], bounds[0])

        # преобразуем т стоп в период (т стоп - т старт)
        params[2] -= params[1]
        params[2] = max(0, params[2])

        token.params = params
        individ.fitness = None
        individ.apply_operator(name='VarFitnessIndivid', grid=grid)
        return individ.fitness

    def _optimize(self, individ):
        target_tokens = list(filter(lambda x: x.mandatory != 0, individ.structure))
        assert len(target_tokens) == 1, 'Individ must have only one target'

        tokens = list(filter(lambda x: x.mandatory == 0, individ.structure))
        assert len(tokens) == 1, 'Individ must have only one optimizing token'

        target_token = target_tokens[0]
        token = tokens[0]
        token.fixator['self'] = True

        grid = self.params['grid']
        constraints = self.params['constraints']

        target = -target_token.value(grid)
        self._set_pulse_start(token, target)

        params = deepcopy(token.params)
        params[2] += params[1]

        res = minimize(self._fitness_wrapper, params,
                       args=(individ, grid, token, constraints), method='Nelder-Mead')

        self._fitness_wrapper(res.x, individ, grid, token, constraints)

    @apply_decorator
    def apply(self, individ, *args, **kwargs):

        self._optimize(individ)


class ImpComplexOptimizerIndivid(GeneticOperatorIndivid):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('grid')

    @staticmethod
    def _get_target_and_complex(individ):
        target_tokens = list(filter(lambda x: x.mandatory != 0, individ.structure))
        assert len(target_tokens) == 1, 'Individ must have only one target'

        complex_tokens = list(filter(lambda x: x.mandatory == 0 and isinstance(x, ImpComplex), individ.structure))
        assert len(complex_tokens) == 1, 'Individ must have only one optimizing ImpComplex token'

        target_token = target_tokens[0]
        complex_token = complex_tokens[0]
        assert not complex_token.fixator['self'], 'ImpComplex already optimized'
        return target_token, complex_token

    def _create_tmp_individ(self, individ):
        grid = self.params['grid']
        target_token, complex_token = self._get_target_and_complex(individ=individ)

        complex_token.init_structure_from_pattern(grid)

        target = -target_token.value(grid)
        target -= target.min()
        target /= target.max()

        tmp_individ = create_tmp_individ(individ, copy(complex_token.structure), target, name='icoi_target')
        return tmp_individ

    def _optimize(self, tmp_individ):
        target_tokens = list(filter(lambda x: x.mandatory != 0, tmp_individ.structure))
        assert len(target_tokens) == 1, 'Individ must have only one target'
        target_token = target_tokens[0]

        tokens = list(filter(lambda x: x.mandatory == 0, tmp_individ.structure))

        grid = self.params['grid']

        delta_left, delta_right = 0.2, 0.2
        for token_idx, token in enumerate(tokens):
            assert not token.fixator['self'], 'token already optimized'
            token_start = token.param(name='Pulse start')
            token_duration = token.param(name='Duration')
            if token_idx == 0:
                constraints = {
                    1: (-delta_left*token_duration, token_start + (1+delta_right)*token_duration),
                    2: (-delta_left*token_duration, token_start + (1+delta_right)*token_duration)
                }
            else:
                prev_token = tokens[token_idx-1]
                try:
                    prev_token_end = grid[prev_token.value(grid) != 0].max()
                except:
                    prev_token_end = prev_token.param(name='Pulse start') + prev_token.param(name='Duration')
                constraints = {
                    1: (prev_token_end-delta_left*token_duration, token_start + (1+delta_right)*token_duration),
                    2: (prev_token_end-delta_left*token_duration, token_start + (1+delta_right)*token_duration)
                }

            bounds = constraints[1]
            new_grid = grid[(grid >= bounds[0]) & (grid <= bounds[1])]

            constants = get_full_constant()
            old_name = deepcopy(target_token.name_)
            old_value = deepcopy(constants[old_name])
            target_token.name_ = 'new_tmp_uniq'
            set_constants(new_tmp_uniq=old_value[(grid >= bounds[0]) & (grid <= bounds[1])])

            new_target = -tmp_individ.value(new_grid)

            target_token.name_ = old_name

            new_target -= new_target.min()

            tmp2_individ = create_tmp_individ(tmp_individ, [token], new_target, name='tmp2_target')
            tmp2_individ.apply_operator(name='ImpSimpleOptimizerIndivid', grid=new_grid, constraints=constraints)


            token.fixator['self'] = True

    @apply_decorator
    def apply(self, individ, *args, **kwargs):

        tmp_individ = self._create_tmp_individ(individ)
        self._optimize(tmp_individ)
        _, complex_token = self._get_target_and_complex(individ)
        complex_token.fixator['self'] = True


class AllImpComplexOptimizerIndivid(GeneticOperatorIndivid):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('grid', 'optimize_id')

    @staticmethod
    def _tokens_fitnesses(individ, tokens):
        fitnesses = []
        for token in tokens:
            individ.add_substructure(token)
            individ.fitness = None
            individ.apply_operator('VarFitnessIndivid')
            fitnesses.append(individ.fitness)
            individ.del_substructure(token)
        return fitnesses

    @staticmethod
    def _get_tmp_individ_with_target_token(individ, complex_token):
        target_tokens = list(filter(lambda token: token.mandatory != 0, individ.structure))
        assert len(target_tokens) == 1, 'There must be only one target token'
        assert not complex_token.fixator['self']

        tmp_individ = individ.clean_copy()
        tmp_individ.structure = copy(target_tokens)
        tmp_individ.add_substructure(complex_token.pattern)
        tmp_individ.apply_operator('TokenFitnessIndivid')
        tmp_individ.del_substructure(complex_token.pattern)

        other_tokens = list(filter(lambda token:
                                   token.mandatory == 0
                                   and token.fixator['self']
                                   and token.fitness < complex_token.pattern.fitness, individ.structure))
        tmp_individ.structure.extend(other_tokens)
        return tmp_individ


    def _optimize_token_params(self, individ, complex_token):
        grid = self.params['grid']
        assert not complex_token.fixator['self']
        extra_tmp_individ = self._get_tmp_individ_with_target_token(individ, complex_token)

        target = -extra_tmp_individ.value(grid)
        target -= target.min()
        target /= target.max()

        tmp_individ = create_tmp_individ(extra_tmp_individ, [complex_token], target, name='aicoi_target')

        tmp_individ.apply_operator(name='ImpComplexOptimizerIndivid', grid=grid)



        # plt.figure('result {}'.format(np.random.randint(0, 1000)))
        # plt.plot(target, label='target')
        # # plt.plot(init_target, label='init_target')
        # plt.plot(complex_token.pattern.value(grid),
        #          label='pattern, w={}'.format(complex_token.pattern.param(name='Frequency')))
        # plt.plot(complex_token.value(grid), label='cimp')
        # for token in complex_token.structure:
        #     plt.plot(token.value(grid)-1)
        # plt.grid(True)
        # plt.legend()
        # plt.show()

    def _choice_tokens_for_optimize1(self, individ):
        optimize_id = self.params['optimize_id']
        choiced_tokens = list(filter(lambda token: token.optimize_id == optimize_id and not token.fixator['self'],
                                     individ.structure))
        for token in choiced_tokens:
            assert not token.fixator['self']
        return choiced_tokens

    @apply_decorator
    def apply(self, individ, *args, **kwargs):

        choiced_tokens = self._choice_tokens_for_optimize1(individ)
        if len(choiced_tokens) == 0:
            return

        for complex_token in choiced_tokens:
            individ.apply_operator('TokenFitnessIndivid')
            self._optimize_token_params(individ, complex_token)
            complex_token.fixator['self'] = True


class ImpComplexOptimizerIndivid2(GeneticOperatorIndivid):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('grid', 'optimize_id')

    def _group_tokens(self, individ):
        target_tokens = list(filter(lambda token: token.mandatory != 0, individ.structure))
        assert len(target_tokens) == 1, 'Individ must have only one target token'
        self.target_token = target_tokens[0]

        self.optimized_tokens = list(filter(lambda token: token.mandatory == 0 and token.fixator['self'],
                                            individ.structure))
        self.non_optimized_complex_tokens = list(filter(lambda token: token.mandatory == 0
                                                         and isinstance(token, ImpComplex)
                                                         and not token.fixator['self']
                                                         and token.optimize_id == self.params['optimize_id'],
                                                         individ.structure))

    def _optimize(self, individ):
        grid = self.params['grid']
        idxs_grid = np.arange(0, len(grid), 1)
        # target = -self.target_token.value(grid)

        delta_left, delta_right = 0.05, 0.05
        for complex_token_idx, complex_token in enumerate(self.non_optimized_complex_tokens):
            assert not complex_token.fixator['self']
            complex_token.fixator['self'] = True
            complex_token.init_structure_from_pattern(grid)
            for token_idx, token in enumerate(complex_token.structure):
                assert not token.fixator['self']
                token.fixator['self'] = True

                token_start = token.param(name='Pulse start')
                token_duration = token.param(name='Duration')
                if token_start + token_duration <= grid.min():
                    continue
                if token_start >= grid.max():
                    continue

                if token_idx != 0:
                    prev_token = complex_token.structure[token_idx-1]
                    try:
                        prev_token_end = grid[prev_token.value(grid) != 0].max()
                    except:
                        prev_token_end = prev_token.param(name='Pulse start') + prev_token.param(name='Duration')
                if token_idx != len(complex_token.structure)-1:
                    next_token = complex_token.structure[token_idx+1]
                    try:
                        next_token_start = grid[next_token.value(grid) != 0].min()
                    except:
                        next_token_start = next_token.param(name='Pulse start')
                if token_idx == 0:
                    constraints = {
                        1: (grid.min()-delta_left*token_duration, next_token_start  + delta_right*token_duration),
                        2: (0, next_token_start - token_start + delta_right*token_duration)
                    }
                elif token_idx == len(complex_token.structure)-1:
                    constraints = {
                        1: (prev_token_end-delta_left*token_duration, grid.max() + delta_right*token_duration),
                        2: (0, grid.max() - token_start + delta_right*token_duration)
                    }
                else:
                    constraints = {
                        1: (prev_token_end-delta_left*token_duration, next_token_start + delta_right*token_duration),
                        2: (0, next_token_start - token_start + delta_right*token_duration)
                    }

                bounds = constraints[1]
                idxs_optimized_grid = idxs_grid[(grid >= bounds[0]) & (grid <= bounds[1])]

                pattern_value1 = token.value(grid[idxs_optimized_grid])
                # if (pattern_value1 == 0).all():
                #     continue
                pattern_value2 = pattern_value1[pattern_value1 > 0]
                pattern_value2 -= pattern_value2.min()

                target = -individ.value(grid)
                target_value = target[idxs_optimized_grid]
                target_value -= target_value.min()

                cor = np.correlate(pattern_value2, target_value, mode='valid')
                pulse_start = grid[idxs_optimized_grid][np.argmax(cor)]
                token.set_param(pulse_start, name='Pulse start')

                params = deepcopy(token.params)
                # params[2] += params[1]

                res = minimize(self._fitness_wrapper, params,
                               args=(individ, token, complex_token, self.optimized_tokens,
                                     self.target_token, grid, idxs_optimized_grid, constraints), method='Nelder-Mead')

                self._fitness_wrapper(res.x, individ, token, complex_token, self.optimized_tokens,
                                      self.target_token, grid, idxs_optimized_grid, constraints)

                # r = np.random.uniform()
                # plt.figure('cor' + str(r))
                # plt.plot(target_value)
                # plt.plot(pattern_value2)
                # plt.plot(cor)
                #
                # plt.figure('res' + str(r))
                # plt.plot(target[idxs_optimized_grid])
                # plt.plot(token.value(grid[idxs_optimized_grid]))
                # plt.plot(target)
                # plt.show()


            self.optimized_tokens.append(complex_token)


    @staticmethod
    def _fitness_wrapper(params, *args):
        (individ, token, complex_token, optimized_tokens,
         target_token, grid, idxs_optimized_grid, constraints) = args

        # вешаем ограничения
        for key, value in constraints.items():
            bounds = value
            assert bounds[0] <= bounds[1], 'MIN > MAX'
            params[key] = min(params[key], bounds[1])
            params[key] = max(params[key], bounds[0])

        # преобразуем т стоп в период (т стоп - т старт)
        # params[2] -= params[1]
        # params[2] = max(0, params[2])
        token.params = params

        tmp_grid = grid[idxs_optimized_grid]
        complex_token.fixator['val'] = False

        tmp_target_token = target_token.copy()
        tmp_target_token.name_ = 'tmp_target_opt'
        tmp_target_token.fixator['val'] = False
        set_constants(tmp_target_opt=-target_token.value(grid)[idxs_optimized_grid])

        tmp_individ = individ.clean_copy()
        tmp_structure = [tmp_target_token]
        tmp_structure.extend(optimized_tokens)
        tmp_structure.append(complex_token)
        tmp_individ.structure = tmp_structure

        tmp_individ.fitness = None
        tmp_individ.apply_operator(name='VarFitnessIndivid', grid=tmp_grid)
        return tmp_individ.fitness

    @apply_decorator
    def apply(self, individ, *args, **kwargs):
        self._group_tokens(individ)
        self._optimize(individ)





