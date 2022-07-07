"""Инициализация популяции"""
# from buildingBlocks.baseline.GeneticOperators import GeneticOperatorIndivid, GeneticOperatorPopulation
from re import L

from sklearn.utils import resample
from buildingBlocks.baseline.BasicEvolutionaryEntities import GeneticOperatorIndivid, GeneticOperatorPopulation
from buildingBlocks.default.geneticOperators.supplementary.Other import check_operators_from_kwargs, apply_decorator
import buildingBlocks.Globals.GlobalEntities as Bg

import numpy as np
from scipy.optimize import minimize
from itertools import product


class InitIndivid(GeneticOperatorIndivid):
    def __init__(self, params=None):
        super().__init__(params=params)
        # print("Init InitIndivid")
        # non_mandatory_tokens = list(filter(lambda token: token.mandatory == 0, self.params['tokens']))
        # if not len(non_mandatory_tokens):
        #     return
        # pz = len(non_mandatory_tokens[0].variable_params)
        # # temp_amplitudes = np.linspace(-10, 10, len(non_mandatory_tokens))
        # temp_amplitudes = np.ones((len(non_mandatory_tokens)))
        # res = {}
        # k_array = 0
        # kars = np.zeros((len(non_mandatory_tokens)), dtype=int)
        # while kars[0] < pz:
        #     rs = np.average((np.sum([non_mandatory_tokens[i].evaluate(np.hstack((temp_amplitudes[i], non_mandatory_tokens[i].variable_params[kars[i]])), self.params['grid']) for i in range(len(non_mandatory_tokens))], axis=0) - Bg.constants['target']) ** 2)
        #     res[tuple(kars)] = rs
        #     for k in range(len(kars) - 1,0,-1):
        #         if k == (len(kars) - 1):
        #             kars[k] += 1
        #         if kars[k] >= pz:
        #             kars[k] = 0
        #             kars[k - 1] += 1

        # res = np.array([key for key in sorted(res, key=res.get)])
        # self.params['ids'] = res


    @apply_decorator
    def apply(self, individ, *args, **kwargs) -> None:
        # individ.apply_operator('MutationIndivid')
        
        count_mandatory_tokens = 0
        mandatory_tokens = list(filter(lambda token: token.mandatory != 0, self.params['tokens']))
        non_mandatory_tokens_all = list(filter(lambda token: token.mandatory == 0, self.params['tokens']))
        if mandatory_tokens:
            individ.add_substructure([token.clean_copy() for token in mandatory_tokens])
            count_mandatory_tokens = len(mandatory_tokens)

        print("test randomize individ")
        number_of_tokens = np.random.choice(np.arange(1, len(non_mandatory_tokens_all) + 1), 1)[0]
        non_mandatory_tokens = np.random.choice(non_mandatory_tokens_all, number_of_tokens)
        print("number of tokens", number_of_tokens, non_mandatory_tokens)
        
        # non_mandatory_tokens_params = np.array([np.array(token.variable_params)[:, args[0], :] for token in non_mandatory_tokens])
        non_mandatory_tokens_params = [np.array(token.variable_params)[:, args[0]] for token in non_mandatory_tokens]
        # non_mandatory_tokens_params = [np.array(token.variable_params)[args[0]] for token in non_mandatory_tokens]
        # non_mandatory_tokens_params = np.array(non_mandatory_tokens_params)

        A = np.array([np.linspace(-10, 10, len(non_mandatory_tokens)) for _ in range(len(non_mandatory_tokens[0].variable_params))])
        A = A.reshape(-1)

        # debug ----------------------------------------------------------

        # print(np.average((np.sum([non_mandatory_tokens[i].evaluate(np.hstack((A[k][i], non_mandatory_tokens[i].variable_params[k][args[0]])), self.params['grid']) for k, i in product(np.arange(len(non_mandatory_tokens[0].variable_params)), np.arange(len(non_mandatory_tokens)))], axis=0) - Bg.constants['target']) ** 2))
        # for k, i in product(np.arange(len(non_mandatory_tokens[0].variable_params)),np.arange(len(non_mandatory_tokens))):
        #     print(non_mandatory_tokens[i].variable_params[k][args[0]])
        ##----------------------------------------------------------------
        # print(np.sum([non_mandatory_tokens[i].evaluate(np.hstack((A[i], non_mandatory_tokens[i].variable_params[args[0]])), self.params['grid']) for i in range(len(non_mandatory_tokens))]))
        # print([non_mandatory_tokens[i].evaluate(np.hstack((A[i], non_mandatory_tokens[i].variable_params[args[0]])), self.params['grid']) for i in range(len(non_mandatory_tokens))])
        # func_podbor = lambda A: np.average((np.sum([non_mandatory_tokens[i].evaluate(np.hstack((A[k * len(non_mandatory_tokens[0].variable_params) + i], non_mandatory_tokens[i].variable_params[k][args[0]])), self.params['grid']) for k, i in product(np.arange(len(non_mandatory_tokens[0].variable_params)), np.arange(len(non_mandatory_tokens)))], axis=0) - Bg.constants['target']) ** 2)

        shp = (len(non_mandatory_tokens), len(non_mandatory_tokens[0].variable_params), 1)
        func_podbor = lambda A: np.average((np.sum([non_mandatory_tokens[token_i].evaluate(np.hstack((A.reshape(shp)[token_i], non_mandatory_tokens_params[token_i])), self.params['grid']) for token_i in np.arange(shp[0])], axis=0) - Bg.constants['target']) ** 2)
        res_amplitude = minimize(func_podbor, A).x
        res_amplitude = res_amplitude.reshape(shp)
        sub = []
        for i in range(len(non_mandatory_tokens)):
            cur_token = non_mandatory_tokens[i].clean_copy()
            # cur_token.params = np.hstack((res_amplitude[i], non_mandatory_tokens[i].variable_params[self.params['ids'][args[0]][i]]))
            tesyt = np.hstack((res_amplitude.reshape(shp)[i], non_mandatory_tokens_params[i]))
            cur_token.params = tesyt
            sub.append(cur_token)
        individ.add_substructure(sub)

        # для оптимизации в лоб
        # current_tokens = []
        # k, j, n = 0, 0, 0
        # while k < individ.max_tokens - count_mandatory_tokens:
        #     current_tokens.append((non_mandatory_tokens[j], non_mandatory_tokens[j].variable_params[n]))
        #     j += 1
        #     if j == len(non_mandatory_tokens):
        #         j = 0
        #         n += 1
        #     k += 1
        # func_podbor = lambda A: np.average((np.sum([current_tokens[i][0].evaluate(np.hstack((A[i], current_tokens[i][1])), self.params['grid']) for i in range(individ.max_tokens - count_mandatory_tokens)], axis=0) - Bg.constants['target']) ** 2)
        # res_amplitude = minimize(func_podbor, np.linspace(-10, 10, individ.max_tokens - count_mandatory_tokens), method="Nelder-Mead").x
        # sub = []
        # for i in range(individ.max_tokens - count_mandatory_tokens):
        #     cur_token = self.params['tokens'][1].clean_copy()
        #     cur_token.params = np.hstack((res_amplitude[i], current_tokens[i][1]))
        #     sub.append(cur_token)
        
        # individ.add_substructure(sub)


class InitPopulation(GeneticOperatorPopulation):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('population_size', 'individ')

    def apply(self, population, *args, **kwargs):
        population.structure = []
        for _ in range(self.params['population_size']):
            new_individ = self.params['individ'].copy()
            new_individ.apply_operator('InitIndivid', _)
            population.structure.append(new_individ)
        return population
