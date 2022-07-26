"""
Contains default inheritors/implementations of baseline classes for Individual.
"""
from copy import copy, deepcopy
from functools import reduce
from buildingBlocks.baseline.BasicEvolutionaryEntities import Individ, Population
import numpy as np


class Equation(Individ):
    """
    An expression that approximates the input data and is used as a basis for building synthetics.
    """
    def __init__(self, structure: list = None,
                 fixator=None,
                 fitness: float = None,
                 used_value: str = 'plus', forms=None,
                 max_tokens: int = None):
        super().__init__(structure=structure, fixator=fixator, fitness=fitness)
        if forms is None:
            forms = []
        self.elitism = False
        self.selected = False

        self.intercept = 0.

        # self.kind = kind
        self.used_value = used_value
        self.forms = forms

        self.max_tokens = max_tokens

    def __getstate__(self):
        # for key in self.__dict__.keys():
        #     if key in ('val',):
        #         self.__dict__[key] = None
        #     if key in ('forms',):
        #         self.__dict__[key] = []
        return self.__dict__

    def __setstate__(self, state: dict):
        self.__dict__ = state

    def copy(self):
        new_copy = deepcopy(self)

        try:
            new_copy.forms = deepcopy(new_copy.forms)
        except:
            pass

        return new_copy

    def clean_copy(self):
        new_copy = type(self)()
        return new_copy

    def formula(self, with_params=False):
        if self.used_value == "plus":
            joinWith = '+'
        else:
            joinWith = '*'
        return joinWith.join(list(map(lambda x: x.name(with_params), self.structure)))

    def value(self, grid):
        fixed_optimized_tokens_in_structure = list(filter(lambda token: token.fixator['self'],
                                                          self.structure))
        if len(fixed_optimized_tokens_in_structure) != 0:
            # if self.used_value == 'plus':
            # вычисляем валуе только от оптимизированных токенов
            val = reduce(lambda val, x: val + x, list(map(lambda x: x.value(grid),
                                                          fixed_optimized_tokens_in_structure)))
            # val -= self.intercept
            # val = val.reshape(grid.shape) 
            return val
        #     elif self.used_value == 'product':
        #         return reduce(lambda val, x: val * x, list(map(lambda x: x.value(t), self.chromo)))
        return np.zeros(grid.shape)

    def get_norm_of_amplitudes(self):
        fixed_optimized_tokens_in_structure = list(filter(lambda token: token.fixator['self'], self.structure))

        if len(fixed_optimized_tokens_in_structure) != 0:
            norm = reduce(lambda norm, ampl: norm + ampl, list(map(lambda token: np.abs(token.param(name='Amplitude')[0]), fixed_optimized_tokens_in_structure)))

            return norm
        
        return 0

    def get_structure(self, *types):
        if not types:
            return copy(self.structure)
        return list(filter(lambda x: x.type in types, self.structure))


class PopulationOfEquations(Population):
    """
    Class with population of Equations.
    """
    def __init__(self, structure: list = None,
                 iterations: int = 0):
        super().__init__(structure=structure)

        self.iterations = iterations
        # self.loggers = [Logger(), Logger()]

    # helper function
    def text_indiv_param(self):
        print('checing poulation')
        for iter, individ in enumerate(self.structure):
            print(iter)

            for token in individ.structure:
                try:
                    print(token, token.param(name='Frequency'))
                except:
                    print(token, token.params.shape)

    def _evolutionary_step(self):
        # self.apply_operator('RegularisationPopulation')
        # self.apply_operator('PeriodicTokensOptimizerPopulation')
        # self.apply_operator('LassoPopulation')
        # self.apply_operator('FitnessPopulation')
        self.text_indiv_param()
        self.apply_operator('UnifierParallelizedPopulation')
        self.text_indiv_param()

        self.apply_operator('RestrictPopulation')
        self.apply_operator('Elitism')
        self.apply_operator('RouletteWheelSelection')
        self.apply_operator('CrossoverPopulation')
        self.apply_operator('MutationPopulation')

    def evolutionary(self):
        self.apply_operator('InitPopulation')
        for n in range(self.iterations):
            print('{}/{}\n'.format(n, self.iterations))
            self._evolutionary_step()
        # self.apply_operator('RegularisationPopulation')
        # self.apply_operator('PeriodicTokensOptimizerPopulation')
        # self.apply_operator('LassoPopulation')
        # self.apply_operator('FitnessPopulation')
        self.apply_operator('UnifierParallelizedPopulation')
        self.apply_operator('RestrictPopulation')

        self.apply_operator('Elitism')
