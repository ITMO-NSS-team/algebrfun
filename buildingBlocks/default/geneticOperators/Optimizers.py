"""
Contains optimizers of parameters of tokens which are need to optimize.
Лучше сюда не заглядывать, тут есть и будут еще сложные логики оптимизации токенов.
"""
from functools import reduce

# from buildingBlocks.baseline.GeneticOperators import GeneticOperatorIndivid, GeneticOperatorPopulation
from buildingBlocks.baseline.BasicEvolutionaryEntities import GeneticOperatorIndivid, GeneticOperatorPopulation
from buildingBlocks.Globals.supplementary.FrequencyProcessor import FrequencyProcessor4TimeSeries as fp
from scipy.optimize import differential_evolution, minimize
import numpy as np
from copy import deepcopy
from multiprocessing import current_process

from buildingBlocks.default.geneticOperators.supplementary.Other import check_or_create_fixator_item, \
    create_tmp_individ, apply_decorator


class PeriodicTokensOptimizerIndivid(GeneticOperatorIndivid):
    """
    Works with periodic simple token objects of the 'Function'class in Individ chromosome.
    Optimizes the parameters of the token for better approximation of input data.
    """

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

    # todo - создать новый индивид с переопределенным таргетом чтобы быстрее считать (для всех типах где
    #  только один токен
    #  оптимизируется, но на самом деле можно и для всех). Создать у индивида клин-копи через инит и __дикт__
    @staticmethod
    def _fitness_wrapper(params, *args):
        # print("DEBUG FOR FITNESS WRAPPER")
        individ, grid, token = args
        # print(token.params)
        # print(params.reshape(grid.shape[0], len(params)//grid.shape[0]))
        token.params = params.reshape(grid.shape[0], len(params)//grid.shape[0])
        individ.fitness = None
        # individ.fixator['VarFitnessIndivid'] = False
        individ.apply_operator(name='VarFitnessIndivid')
        return individ.fitness

    @staticmethod
    def _find_target_token(individ):
        target_tokens = list(filter(lambda token: token.mandatory != 0, individ.structure))
        assert len(target_tokens) == 1, 'There must be only one target token'
        return target_tokens[0]

    def _optimize_token_params(self, individ, token):
        grid = self.params['grid'] #!!!!!
        # неоптимизированные токены в валуе не считаются
        target = -individ.value(grid)
        # print("debug target", target)
        # центрирование и нормализация (fitness - дисперсия, так что центрирование ничего не меняет)
        # target -= target.mean()
        # target /= np.abs(target).max()
        target -= target.min()
        # target /= target.max()

        tmp_individ = create_tmp_individ(individ, [token], target)
        token.fixator['self'] = True

        freq = fp.choice_freq_for_summand(grid, target-target.mean(),
                                          number_selecting=5, number_selected=5, token_type='seasonal')
        if freq is None: #TODO: сделать проверку присутствия нужного токена в неком пуле, чтобы избежать повторной оптимизаци
            individ.structure.remove(token) # del hopeless token and out
        else:
            eps = self.params['eps']
            bounds = deepcopy(token.get_descriptor_foreach_param(descriptor_name='bounds'))
            # print("optimizers_token", token, "optimizers_bounds", bounds)
            # print("getting freq", freq, eps)
            index = token.get_key_use_params_description(descriptor_name='name',
                                                        descriptor_value='Frequency')
            new_bounds = []
            for freq_i in freq[0]:
                bounds[index] = (freq_i * (1 - eps), freq_i * (1 + eps))
                new_bounds.extend(bounds)
            # freq_bounds = [(freq_i * (1 - eps), freq_i * (1 + eps)) for freq_i in freq[0]]
            # del bounds[index]
            # bounds[index:index] = freq_bounds
            # print("bounds for dif evolution", new_bounds)
            x0 = deepcopy(token.params)
            # shp = x0.shape
            # x0 = x0.reshape(-1)
            for i in range(x0.shape[0]):
                x0[i][token.get_key_use_params_description(descriptor_name='name',
                                                    descriptor_value='Frequency')] = freq[0][i]
            # print("showing x0", x0)
            # x0[0] = 1.
            if self.params['optimizer'] == 'DE':
                # print("x0 not")
                res = differential_evolution(self._fitness_wrapper, new_bounds,
                                             args=(tmp_individ, grid, token),
                                             popsize=self.params['popsize'])
                # print("differential evolution shape:", res)
            else:
                # print("x0", x0.reshape(-1))
                res = minimize(self._fitness_wrapper,  x0.reshape(-1),
                               args=(tmp_individ, grid, token)) # либо переделать функцию _fitness_wrapper, чтобы x0 не был одномерным массивом
            token.params = res.x.reshape(grid.shape[0], len(res.x)//grid.shape[0])
            # print(token.params)
        # individ.change_all_fixes(False)
        individ.structure = individ.structure

    def _choice_tokens_for_optimize(self, individ):
        optimize_id = self.params['optimize_id']
        choiced_tokens = list(filter(lambda token: token.optimize_id == optimize_id and not token.fixator['self'],
                                     individ.structure))
        return choiced_tokens

    @apply_decorator
    def apply(self, individ, *args, **kwargs) -> None:
        choiced_tokens = self._choice_tokens_for_optimize(individ)
        for token in choiced_tokens:
            self._optimize_token_params(individ, token)


class PeriodicExtraTokensOptimizerIndivid(GeneticOperatorIndivid): # todo надо переделать под улучшение с доп индивидом
    """
    Works with periodic simple token objects of the 'Function'class in Individ chromosome.
    Optimizes the parameters of the token for better approximation of input data.
    """

    def __init__(self, params=None):
        if params is None:
            params = {}
        add_params = {
            'optimize_id': None,
        }
        for key, value in add_params.items():
            if key not in params.keys():
                params[key] = value
        super().__init__(params=params)
        self._check_params('grid', 'optimize_id')

    @staticmethod
    def _fitness_wrapper(params, *args):
        individ, grid, tokens = args

        start = 0
        for token in tokens:
            token.params = params[start: start + token._number_params]
            start += token._number_params

        individ.fitness = None
        individ.apply_operator(name='VarFitnessIndivid')
        return individ.fitness

    def _choice_tokens_for_optimize(self, individ):
        optimize_id = self.params['optimize_id']
        choiced_tokens = list(filter(lambda token: token.optimize_id == optimize_id,
                                     individ.structure))
        return choiced_tokens

    def _optimize_token_params(self, individ, tokens):
        grid = self.params['grid']

        x0 = []
        for token in tokens:
            token.fixator['self'] = True
            x0.extend(list(deepcopy(token.params)))
        x0 = np.array(x0)

        res = minimize(self._fitness_wrapper,  x0,
                       args=(individ, grid, tokens))
        # print(self._fitness_wrapper([1, 1, 1], individ, grid, target, token))
        start = 0
        for token in tokens:
            token.params = res.x[start: start + token._number_params]
            start += token._number_params

        individ.structure = individ.structure

    @apply_decorator
    def apply(self, individ, *args, **kwargs) -> None:
        choiced_tokens = self._choice_tokens_for_optimize(individ)
        if len(choiced_tokens) == 0:
            return
        self._optimize_token_params(individ, choiced_tokens)


class PeriodicInProductTokensOptimizerIndivid(GeneticOperatorIndivid): # todo испрвить возможный баг с фиксациями токенов
    """
    Works with periodic simple token objects of the 'Function'class in Individ chromosome.
    Optimizes the parameters of the token for better approximation of input data.
    """

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
        individ, grid, tokens = args
        params = np.array_split(params, len(tokens))
        for idx, token in enumerate(tokens):
            #todo вынести парамс_идх в *аргс чтобы не пересчитывать каждый раз
            params_idxs = [i for i in range(len(token.params)) if i !=
                           token.get_key_use_params_description(descriptor_name='name',
                                                                descriptor_value='Frequency')]
            token.params[params_idxs] = params[idx]
            token.fixator['val'] = False
            # token.params = params[idx]
            # if idx != 0:
            #     token.set_param(1, name='Amplitude') #TODO: Надо что то сделать чтобы не занулялись амплитуды при поиске
        # для пересчета фитнеса индивида
        individ.fitness = None
        individ.apply_operator(name='VarFitnessIndivid')
        return individ.fitness #TODO: нужно сверху наложить штраф на маленькие амплитуды для использования ДЕ

    def _preprocessing_product_tokens(self, individ):
        grid = self.params['grid']
        choiced_subtokens, complex_tokens_with_choiced_subtokens = self._choice_tokens_for_optimize(
            individ, ret_complex_tokens_with_choiced_subtokens=True)
        if len(choiced_subtokens) == 1:
            assert len(complex_tokens_with_choiced_subtokens) == 1
            current_complex_token = complex_tokens_with_choiced_subtokens[0]
            choiced_subtoken = choiced_subtokens[0]
            target_chromo = list(filter(lambda token: token != current_complex_token and token.mandatory != 0,
                                        individ.structure))
            #TODO: можно заменить проверку на token.fix
            current_complex_token_fix_subtokens = list(filter(lambda token: token != choiced_subtoken,
                                                              current_complex_token.subtokens))
            current_complex_token_value = current_complex_token.param(name='Amplitude')*reduce(
                lambda val, x: val * x, list(map(lambda x: x.value(grid), current_complex_token_fix_subtokens)))
            current_complex_token_freqs = fp.find_freq_for_summand(grid, current_complex_token_value,
                                                                   number_selecting=len(current_complex_token_fix_subtokens),
                                                                   number_selected=len(current_complex_token_fix_subtokens))
            optimization_info = []
            for target_idx, target_token in enumerate(target_chromo):
                target = -target_token.value(grid)
                recommended_freqs = fp.find_freq_for_multiplier(grid, target,
                                                                current_complex_token_freqs, max_len=10)
                if len(recommended_freqs) == 0:
                    continue
                for recommended_freq in recommended_freqs:
                    optimization_info.append([target_idx, target_token, *recommended_freq])

            if len(optimization_info) == 0:
                current_complex_token.del_subtoken(choiced_subtoken)
                return

            optimization_case = optimization_info[np.random.randint(len(optimization_info))]
            target_idx, target_token, current_complex_token_freq, lower_freq, higher_freq, diff_freq = optimization_case
            W_max = 1/np.mean(grid[1:] - grid[:-1])
            # если нижняя разностная частота уходит в ноль, значит несущая частота и разностная частота равны,
            # и мы должны домножить токен-случай с таким сходством на выбранный токен, и оптимизировать фазы
            if lower_freq <= 0.01*W_max:
                if type(target_token) in self.params['complex_tokens_types']:
                    target_token.add_token(choiced_subtoken.copy())
                else:
                    new_complex_token = type(current_complex_token)()
                    new_complex_token.subtokens = [target_token, choiced_subtoken.copy()]
                    individ.structure[individ.structure.index(target_token)] = new_complex_token
                return current_complex_token_freq

            return diff_freq
        else:
            assert len(choiced_subtokens) == 1

    def _optimize_token_params(self, individ, tokens: list):
        grid = self.params['grid']
        individ.apply_operator(name='LassoIndivid1Target', use_lasso=False)
        freq = self._preprocessing_product_tokens(individ)
        if freq is None: #TODO: сделать проверку присутствия нужного токена в неком пуле, чтобы избежать повторной оптимизаци
            for token in tokens:
                for ind_token in individ.structure:
                    try:
                        ind_token.del_subtoken(token)
                    except:
                        pass
        else:
            tokens, complex_tokens_with_choiced_subtokens = \
                self._choice_tokens_for_optimize(individ,
                                                 ret_complex_tokens_with_choiced_subtokens=True)
            for token in complex_tokens_with_choiced_subtokens:
                token.fixator['cache'] = False
            eps = self.params['eps']
            bounds = []
            x0 = []
            for token in tokens:
                token.set_param(freq, name='Frequency')
                token.set_param(1., name='Amplitude') #TODO: убрать из оптимизируемых параметров частоту и одну из амплитуд
                token.set_param(0., name='Phase')

            tokens = tokens[:1]
            for token in tokens:
                bounds_token = deepcopy(token.get_descriptor_foreach_param(descriptor_name='bounds'))
                # bounds_token[token.get_key_use_params_description(descriptor_name='name',
                #                                                   descriptor_value='Frequency')] = (freq * (1 - eps),
                #                                                                                     freq * (1 + eps))
                del bounds_token[token.get_key_use_params_description(descriptor_name='name',
                                                                      descriptor_value='Frequency')]
                bounds.extend(bounds_token)

                x0_token = list(deepcopy(token.params))
                # x0_token[token.get_key_use_params_description(descriptor_name='name',
                #                                               descriptor_value='Frequency')] = freq
                del x0_token[token.get_key_use_params_description(descriptor_name='name',
                                                                  descriptor_value='Frequency')]
                x0.extend(x0_token)

            x0 = np.array(x0)

            if self.params['optimizer'] == 'DE':
                res = differential_evolution(self._fitness_wrapper, bounds,
                                             args=(individ, grid, tokens),
                                             popsize=self.params['popsize'])
            else:
                res = minimize(self._fitness_wrapper,  x0,
                               args=(individ, grid, tokens))

            # print(res)
            answer = np.array_split(res.x, len(tokens)) #todo не подходит для токенов с разным количеством параметров
            for idx, token in enumerate(tokens):
                params_idxs = [i for i in range(len(token.params)) if i !=
                               token.get_key_use_params_description(descriptor_name='name',
                                                                    descriptor_value='Frequency')]
                token.params[params_idxs] = answer[idx]
                token.fixator['val'] = False
                # token.params = answer[idx]
                token.fixator['self'] = True
            for token in complex_tokens_with_choiced_subtokens:
                token.fixator['cache'] = True
        individ.structure = individ.structure

        try:
            individ.forms.append(type(self).__name__ + individ.formula() + '<---' + current_process().name)
        except:
            pass

    def _choice_tokens_for_optimize(self, individ, ret_complex_tokens_with_choiced_subtokens=False):
        optimize_id = self.params['optimize_id']
        complex_tokens_in_chromo = list(filter(lambda token: type(token) in self.params['complex_tokens_types'],
                                                individ.structure))
        choiced_subtokens = list(filter(lambda subtoken: subtoken.optimize_id == optimize_id and not subtoken.fixator['self'],
                                        [subtoken for token in complex_tokens_in_chromo
                                         for subtoken in token.subtokens]))
        # Зануляем амплитуды токенов чтобы не мешали оптимизирововать друг друга по порядку
        for token in choiced_subtokens:
            token.set_param(0., name='Amplitude')
        if ret_complex_tokens_with_choiced_subtokens:
            complex_tokens_with_choiced_subtokens = []
            for token in complex_tokens_in_chromo:
                for subtoken in token.subtokens:
                    if subtoken in choiced_subtokens:
                        complex_tokens_with_choiced_subtokens.append(token)
            # complex_tokens_with_choiced_subtokens = list(filter(lambda token:
            #                                                     not set(token.subtokens).isdisjoint(
            #                                                         set(choiced_subtokens)),
            #                                                     complex_tokens_in_chromo))
            return choiced_subtokens, complex_tokens_with_choiced_subtokens
        return choiced_subtokens

    @apply_decorator
    def apply(self, individ, *args, **kwargs) -> None:
        choiced_tokens = self._choice_tokens_for_optimize(individ)
        self._optimize_token_params(individ, tokens=choiced_tokens)


class TrendDiscreteTokensOptimizerIndivid(PeriodicTokensOptimizerIndivid):
    def __init__(self, params=None):
        super().__init__(params=params)

    def _optimize_token_params(self, individ, token):
        grid = self.params['grid']
        # неоптимизированные токены в валуе не считаются
        target = -individ.value(grid)
        # центрирование и нормализация (fitness - дисперсия, так что центрирование ничего не меняет)
        # target -= target.mean()
        # target /= np.abs(target).max()
        target -= target.min()
        target /= target.max()

        tmp_individ = create_tmp_individ(individ, [token], target)
        token.fixator['self'] = True

        freq = fp.choice_freq_for_summand(grid, target-target.mean(),
                                          number_selecting=5, number_selected=5, token_type='trend')
        if freq is None: #TODO: сделать проверку присутствия нужного токена в неком пуле, чтобы избежать повторной оптимизаци
            individ.structure.remove(token) # del hopeless token and out
        else:
            bounds = deepcopy(token.get_descriptor_foreach_param(descriptor_name='bounds'))
            x0 = deepcopy(token.params)

            if self.params['optimizer'] == 'DE':
                res = differential_evolution(self._fitness_wrapper, bounds,
                                             args=(individ, grid, token),
                                             popsize=self.params['popsize'])
            else:
                res = minimize(self._fitness_wrapper,  x0,
                               args=(tmp_individ, grid, token))
            token.params = res.x
            token.fixator['self'] = True
        individ.structure = individ.structure

    @apply_decorator
    def apply(self, individ, *args, **kwargs):
        choiced_tokens = self._choice_tokens_for_optimize(individ)
        for token in choiced_tokens:
            self._optimize_token_params(individ, token)


class TrendTokensOptimizerIndivid(PeriodicExtraTokensOptimizerIndivid):
    def __init__(self, params=None):
        super().__init__(params=params)


# class UniversalOptimizerIndivid(GeneticOperatorIndivid):
#     def __init__(self, params=None):
#         if params is None:
#             params = {}
#         add_params = {
#             'optimize_id': None,
#         }
#         for key, value in add_params.items():
#             if key not in params.keys():
#                 params[key] = value
#         super().__init__(params=params)
#         self._check_params('grid', 'optimize_id')
#
#     def



class PeriodicTokensOptimizerPopulation(GeneticOperatorPopulation):
    def __init__(self, params=None):
        super().__init__(params=params)

    def apply(self, population, *args, **kwargs):
        for individ in population.structure:
            # individ.apply_operator('TrendTokensOptimizerIndivid')
            individ.apply_operator('TrendDiscreteTokensOptimizerIndivid')
            # individ.apply_operator('ImpComplexOptimizerIndivid2')
            individ.apply_operator('ImpComplexDiscreteTokenParamsOptimizer')
            # individ.apply_operator('AllImpComplexOptimizerIndivid')
            individ.apply_operator('PeriodicTokensOptimizerIndivid')
            # individ.apply_operator('PeriodicExtraTokensOptimizerIndivid')
        return population




