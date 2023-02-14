import numpy as np
from copy import deepcopy, copy
from functools import reduce

from .base import Individ
from .base import Population

class Equation(Individ):
    """
    An eqaution that approximates the process of input data and is used as a basis for building synthetics.
    """
    def __init__(self, structure: list = None,
                 fitness: float = None,
                 used_value: str = 'plus', forms=None,
                 max_tokens: int = None):
        super().__init__(structure=structure, fitness=fitness)
        if forms is None:
            forms = []
        self.elitism = False
        self.selected = False

        self.intercept = 0.

        # self.kind = kind
        self.used_value = used_value
        self.forms = forms

        self.max_tokens = max_tokens
        self.type_ = "Equation"

    
    def __getstate__(self):
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
        # fixed_optimized_tokens_in_structure = list(filter(lambda token: token.fixator['self'], self.structure))
        fixed_optimized_tokens_in_structure = self.structure
        # print("checking value in equation", fixed_optimized_tokens_in_structure)
        if len(fixed_optimized_tokens_in_structure) != 0:
            # if self.used_value == 'plus':
            # вычисляем валуе только от оптимизированных токенов
            # print('Mean values of f-ing tokens', [np.mean(token.value(grid)) for token in fixed_optimized_tokens_in_structure])
            val = reduce(lambda val, x: val + x, list(map(lambda x: x.value(grid),
                                                          fixed_optimized_tokens_in_structure)))
            # val -= self.intercept
            # val = val.reshape(grid.shape) 
            # print("returned value",  val)
            return val
        #     elif self.used_value == 'product':
        #         return reduce(lambda val, x: val * x, list(map(lambda x: x.value(t), self.chromo)))
        return np.zeros((grid.shape[-1],))

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
        self.type_ = "PopulationOfEquations"

    def _evolutionary_step(self, *args):
        self.apply_operator('TokenParametersOptimizerPopulation')
        for individ in self.structure:
            individ.apply_operator('TokenFitnessIndivid')
            individ.apply_operator('FilterIndivid')
            individ.apply_operator('LRIndivid')
        self.apply_operator("Fitnesspopulation")
        

    def evolutionary(self, *args):
        self.apply_operator('InitPopulation')
        for n in range(1):
            print('{}/{}\n'.format(n, self.iterations))
            self._evolutionary_step()

def _methods_decorator(method):
    def wrapper(*args, **kwargs):
        self = args[0]
        self.change_all_fixes(False)
        return method(*args, **kwargs)
    return wrapper
