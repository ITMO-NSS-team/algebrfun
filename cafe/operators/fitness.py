"""Реализации функций вычисления фитнеса индивида, вклада в фитнес токена в индивиде, и обощение для популяции"""

# from buildingBlocks.baseline.GeneticOperators import GeneticOperatorIndivid, GeneticOperatorPopulation
from copy import copy, deepcopy
import numpy as np

from .base import GeneticOperatorIndivid
from .base import GeneticOperatorPopulation
from .base import apply_decorator


class VarFitnessIndivid(GeneticOperatorIndivid):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('grid')

    # @apply_decorator
    def apply(self, individ, *args, **kwargs) -> None:
        lmd = 0.4
        individ.fitness = np.linalg.norm(individ.value(self.params['grid'])) + lmd * np.linalg.norm(individ.get_sm_amplitudes(), ord=1)


class TokenFitnessIndivid(GeneticOperatorIndivid):
    def __init__(self, params=None):
        super().__init__(params=params)
        # self._check_params()  

    @staticmethod
    def _label_tokens(individ):
        target_tokens = list(filter(lambda token: token.mandatory != 0, individ.structure))

        other_tokens = list(filter(lambda token: token.mandatory == 0, individ.structure))
        if not other_tokens:
            return

        tmp_individ = individ.clean_copy()        
        tmp_individ.structure = copy(target_tokens)

        for token in other_tokens:
            # assert token.fixator['self'], 'token must be optimized'
            tmp_individ.add_substructure(token)
            tmp_individ.fitness = None
            tmp_individ.apply_operator('VarFitnessIndivid')
            token.fitness = deepcopy(tmp_individ.fitness)
            tmp_individ.del_substructure(token)

        fits = list(map(lambda token: token.fitness, other_tokens))
        sorted_idxs = np.argsort(fits)

        other_tokens = [other_tokens[i] for i in sorted_idxs]

        individ.structure = target_tokens
        individ.add_substructure(other_tokens)

    @apply_decorator
    def apply(self, individ, *args, **kwargs):
        self._label_tokens(individ)


class FitnessPopulation(GeneticOperatorPopulation):
    def __init__(self, params=None):
        super().__init__(params=params)

    def apply(self, population, *args, **kwargs):
        for individ in population.structure:
            individ.apply_operator('VarFitnessIndivid')
        return population

