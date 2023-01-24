"""Инициализация популяции"""
# from buildingBlocks.baseline.GeneticOperators import GeneticOperatorIndivid, GeneticOperatorPopulation
from email import iterators
from re import L

from sklearn.utils import resample
from buildingBlocks.baseline.BasicEvolutionaryEntities import DifferentialToken, DifferentialTokenConstant, GeneticOperatorIndivid, GeneticOperatorPopulation
from buildingBlocks.default.EvolutionEntities import DEquation, Equation
from buildingBlocks.default.geneticOperators.supplementary.Other import check_operators_from_kwargs, apply_decorator
import buildingBlocks.Globals.GlobalEntities as Bg
from buildingBlocks.Globals.GlobalEntities import set_constants, get_full_constant
from buildingBlocks.default.EvolutionEntities import PopulationOfEquations, PopulationOfDEquations

import numpy as np
from scipy.optimize import minimize
from itertools import product
import random


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
        # инициализация рандомных структур диф.уравнений
        # if individ.type_ == "DEquation":
        #     constants = get_full_constant()
        #     der_set = constants['pul_mtrx']
        #     constant_token = list(filter(lambda curtoken: curtoken.type == "Constant", self.params['tokens']))
        #     sin_token = list(filter(lambda curtoken: curtoken.name_ == "Sin", self.params['tokens']))
        #     number_of_temps = np.random.randint(1, len(der_set) + 1)
        #     selected_temps = np.random.choice(np.arange(number_of_temps), number_of_temps, replace=False)
        #     target_eq = args[1].structure[0].structure[0]
        #     # constant_token[0].params[0] = np.array([np.random.choice(constant_token[0].variable_params[0])])
        #     constant_token[0].params[0] = np.array([1.0])
        #     sin_token[0].params = np.array([[1.0, 1.0, 0.0]])
        #     sin_token[0].fixator['self'] = True
        #     # target_eq.structure = [constant_token[0].copy()]
        #     target_eq.structure = [sin_token[0].copy()]
        #     const_term = DifferentialTokenConstant(number_params=2, params_description={0: dict(name='const param'), 1: dict(name="Term")}, params=np.array([target_eq, constants['target']], dtype=object), name_="DifferentialToken")
        #     # selected_temps.append(constants['target'].term_id)
        #     # sub = [DifferentialToken(number_params=2, params_description={0: dict(name='Close algebr equation'), 1: dict(name="Term")}, params=np.array([random.choice(args[1].structure[current_temp].structure), der_set[current_temp]], dtype=object), name_="DifferencialToken") for current_temp in selected_temps]
        #     sub = []
        #     description = {0: dict(name='Close algebr equation'), 1: dict(name="Term")}
        #     for current_temp in selected_temps:
        #         target_eq = args[1].structure[current_temp].structure[0].copy()
        #         # target_eq.structure = [target_eq.structure[0]]
        #         target_eq.structure = [constant_token[0].copy()]
        #         current_token = DifferentialToken(number_params=2, params_description=description, params=np.array([target_eq, der_set[current_temp]], dtype=object), name_="DifferencialToken")
        #         sub.append(current_token)
        #     # sub = [DifferentialToken(number_params=2, params_description={0: dict(name='Close algebr equation'), 1: dict(name="Term")}, params=np.array([random.choice(args[1].structure[current_temp].structure), der_set[current_temp]], dtype=object), name_="DifferencialToken") ]
        #     sub.append(const_term)
        #     individ.add_substructure(sub)
        #     all_tokens = [DifferentialToken(number_params=2, params_description={0: dict(name='Close algebr equation'), 1: dict(name="Term")}, params=np.array([random.choice(args[1].structure[current_temp].structure), der_set[current_temp]], dtype=object), name_="DifferencialToken") for current_temp in np.arange(len(der_set))]
        #     set_constants(tokens=all_tokens)
        #     return

        if individ.type_ == "DEquation":
            constants = get_full_constant()
            der_set = constants['pul_mtrx']
            constant_token = list(filter(lambda curtoken: curtoken.type == "Constant", self.params['tokens']))
            constant_token[0].params[0] = 1
            target_eq = args[1].structure[0].structure[0].copy()
            target_eq.structure = [constant_token[0].copy()]
            const_term = DifferentialTokenConstant(number_params=2, params_description={0: dict(name='const param'), 1: dict(name="Term")}, params=np.array([target_eq, constants['target']], dtype=object), name_="DifferentialToken")
            sub = []
            for trm in der_set:
                # target_eq = args[1].structure[0].structure[0].copy()
                # target_eq.structure = [constant_token[0].copy()]
                current_tkn = DifferentialToken(number_params=2, params_description={0: dict(name='Close algebr equation'), 1: dict(name="Term")}, params=np.array([target_eq, trm], dtype=object), name_="DifferencialToken")
                sub.append(current_tkn)
            sub.append(const_term)
            individ.add_substructure(sub)
            return

        count_mandatory_tokens = 0
        non_mandatory_tokens_all = self.params['tokens']

        # print("test randomize individ")
        number_of_tokens = np.random.choice(np.arange(1, len(non_mandatory_tokens_all) + 1), 1)[0]
        non_mandatory_tokens = np.random.choice(non_mandatory_tokens_all, number_of_tokens)
        # print("number of tokens", number_of_tokens, non_mandatory_tokens)
        
        # non_mandatory_tokens_params = np.array([np.array(token.variable_params)[:, args[0], :] for token in non_mandatory_tokens])
        free_number = min(list(map(lambda elem: np.array(elem.variable_params).shape[1], non_mandatory_tokens)))
        selectors = np.random.choice(np.arange(free_number), number_of_tokens)
        non_mandatory_tokens_params = [np.array(token.variable_params)[:, int(selectors[iter])] for iter, token in enumerate(non_mandatory_tokens)]
        # non_mandatory_tokens_params = [np.array(token.variable_params)[:, args[0]] for iter, token in enumerate(non_mandatory_tokens)]
        # non_mandatory_tokens_params = [np.array(token.variable_params)[args[0]] for token in non_mandatory_tokens]
        # non_mandatory_tokens_params = np.array(non_mandatory_tokens_params)

        # A = np.array([np.linspace(-10, 10, len(non_mandatory_tokens)) for _ in range(len(non_mandatory_tokens[0].variable_params))])
        A = np.array([np.random.choice(np.arange(-10, 10), len(non_mandatory_tokens)) for _ in range(len(non_mandatory_tokens[0].variable_params))])
        A = A.reshape(-1)

        shp = (len(non_mandatory_tokens), len(non_mandatory_tokens[0].variable_params), 1)
        # func_podbor = lambda A: np.average((np.sum([non_mandatory_tokens[token_i].evaluate(np.hstack((A.reshape(shp)[token_i], non_mandatory_tokens_params[token_i])), self.params['grid']) for token_i in np.arange(shp[0])], axis=0) - Bg.constants['target']) ** 2) # ? нужно переписать под уравнения ? 
        # res_amplitude = minimize(func_podbor, A).x
        # res_amplitude = res_amplitude.reshape(shp)
        sub = []
        for i in range(len(non_mandatory_tokens)):
            cur_token = non_mandatory_tokens[i].clean_copy()
            # cur_token.params = np.hstack((res_amplitude[i], non_mandatory_tokens[i].variable_params[self.params['ids'][args[0]][i]]))
            # tesyt = np.hstack((res_amplitude.reshape(shp)[i], non_mandatory_tokens_params[i]))
            if cur_token.type == "Constant":
                tesyt = non_mandatory_tokens_params[i]
            else:
                tesyt = np.hstack((A.reshape(shp)[i], non_mandatory_tokens_params[i]))
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
            new_individ.apply_operator('InitIndivid', _, args[0])
            population.structure.append(new_individ)
        return population

class InitSubPopulations(GeneticOperatorPopulation):
    def __init__(self, params):
        super().__init__(params=params)
        # self._check_params('population_size')

    def apply(self, population, *args, **kwargs):
        constants = get_full_constant()
        der_set = constants['pul_mtrx']

        population.structure = []
        for elem in der_set:
            tmp_population = PopulationOfEquations(iterations=population.iterations)
            tmp_population.apply_operator("InitPopulation", population)
            tmp_population.owner_id = elem.term_id
            population.structure.append(tmp_population)
        tmp_population = PopulationOfDEquations(iterations=population.iterations)
        tmp_population.apply_operator("InitPopulation", population)
        # constants['best_individ'] = tmp_population.structure[0]
        set_constants(best_individ=tmp_population.structure[0])
        population.structure.append(tmp_population)
        return population
