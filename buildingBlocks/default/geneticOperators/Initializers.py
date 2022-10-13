"""Инициализация популяции"""
# from buildingBlocks.baseline.GeneticOperators import GeneticOperatorIndivid, GeneticOperatorPopulation
from email import iterators
from re import L

from sklearn.utils import resample
from buildingBlocks.baseline.BasicEvolutionaryEntities import DifferentialToken, GeneticOperatorIndivid, GeneticOperatorPopulation
from buildingBlocks.default.EvolutionEntities import DEquation, Equation
from buildingBlocks.default.geneticOperators.supplementary.Other import check_operators_from_kwargs, apply_decorator
import buildingBlocks.Globals.GlobalEntities as Bg
from buildingBlocks.Globals.GlobalEntities import set_constants, get_full_constant
from buildingBlocks.default.EvolutionEntities import PopulationOfEquations, PopulationOfDEquations

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
        if individ.type_ == "DEquation":
            constants = get_full_constant()
            der_set = constants['pul_mtrx']
            number_of_temps = len(der_set)
            selected_temps = np.random.choice(np.arange(number_of_temps), number_of_temps)
            sub = [DifferentialToken(number_params=2, params_description={0: dict(name='Close algebr equation'), 1: dict(name="Term")}, params=np.array([Equation(max_tokens=10), der_set[current_temp]], dtype=object), name_="DifferencialToken") for current_temp in selected_temps]
            individ.add_substructure(sub)
            return

        count_mandatory_tokens = 0
        mandatory_tokens = list(filter(lambda token: token.mandatory != 0, self.params['tokens']))
        non_mandatory_tokens_all = list(filter(lambda token: token.mandatory == 0, self.params['tokens']))
        if mandatory_tokens:
            individ.add_substructure([token.clean_copy() for token in mandatory_tokens])
            count_mandatory_tokens = len(mandatory_tokens)

        # print("test randomize individ")
        number_of_tokens = np.random.choice(np.arange(1, len(non_mandatory_tokens_all) + 1), 1)[0]
        non_mandatory_tokens = np.random.choice(non_mandatory_tokens_all, number_of_tokens)
        # print("number of tokens", number_of_tokens, non_mandatory_tokens)
        
        # non_mandatory_tokens_params = np.array([np.array(token.variable_params)[:, args[0], :] for token in non_mandatory_tokens])
        non_mandatory_tokens_params = [np.array(token.variable_params)[:, args[0]] for token in non_mandatory_tokens]
        # non_mandatory_tokens_params = [np.array(token.variable_params)[args[0]] for token in non_mandatory_tokens]
        # non_mandatory_tokens_params = np.array(non_mandatory_tokens_params)

        A = np.array([np.linspace(-10, 10, len(non_mandatory_tokens)) for _ in range(len(non_mandatory_tokens[0].variable_params))])
        A = A.reshape(-1)

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


class InitPopulation(GeneticOperatorPopulation):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('population_size', 'individ')

    def apply(self, population, *args, **kwargs):
        population.structure = []
        for _ in range(self.params['population_size']):
            new_individ = self.params['individ'].copy()
            if population.type_ == "PopulationOfDEquation":
                new_individ = DEquation(max_tokens=10)
            new_individ.apply_operator('InitIndivid', _)
            population.structure.append(new_individ)
        return population

class InitSubPopulations():
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('population_size', 'individ')

    def apply(self, population, *args, **kwargs):
        constants = get_full_constant()
        der_set = constants['pul_mtrx']
        population.structure = []
        for i, elem in der_set:
            tmp_population = PopulationOfEquations(iterations=population.iterations)
            tmp_population.apply_operator("InitPopulation")
            population.structure.append(tmp_population)
        tmp_population = PopulationOfDEquations(iterations=population.iterations)
        tmp_population.apply_operator("InitPopulation")
        population.structure.append(tmp_population)
        return population
