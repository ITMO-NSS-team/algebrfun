"""Реализации функций вычисления фитнеса индивида, вклада в фитнес токена в индивиде, и обощение для популяции"""

# from buildingBlocks.baseline.GeneticOperators import GeneticOperatorIndivid, GeneticOperatorPopulation
from copy import copy, deepcopy
from multiprocessing import current_process

from buildingBlocks.baseline.BasicEvolutionaryEntities import GeneticOperatorIndivid, GeneticOperatorPopulation
from functools import reduce
import numpy as np

from buildingBlocks.default.geneticOperators.supplementary.Other import check_or_create_fixator_item, \
    check_operators_from_kwargs, apply_decorator


class VarFitnessIndivid(GeneticOperatorIndivid):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('grid')

    # @apply_decorator
    def apply(self, individ, *args, **kwargs) -> None:
        # vec = reduce(lambda val, token: val + token,
        #              list(map(lambda token: token.value(self.params['grid']), individ.structure)))
        # individ.fitness = np.linalg.norm(vec)
        target_token = list(filter(lambda token: token.mandatory != 0, individ.structure))[0]
        ampl_norm = individ.get_norm_of_amplitudes()
        # lmd = 10000000
        # lmd = 10000
        lmd = 1e-100000000
        vec = individ.value(self.params['grid']) + lmd * ampl_norm
        # vec = individ.value(self.params['grid'])
        individ.fitness = np.var(vec)/np.var(target_token.value(self.params['grid']))
        # individ.fitness = np.abs(vec - vec.mean()).mean() 


class TokenFitnessIndivid(GeneticOperatorIndivid):
    def __init__(self, params=None):
        super().__init__(params=params)
        self._check_params()

    @staticmethod
    def _label_tokens(individ):
        target_tokens = list(filter(lambda token: token.mandatory != 0, individ.structure))
        assert len(target_tokens) == 1, 'Individ must have only one target token'

        other_tokens = list(filter(lambda token: token.mandatory == 0 and token.fixator['self'], individ.structure))
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

        tmp_fixator = deepcopy(individ.fixator)
        # sorted(other_tokens, key=lambda token: token.fitness)
        fits = list(map(lambda token: token.fitness, other_tokens))
        sorted_idxs = np.argsort(fits)

        other_tokens = [other_tokens[i] for i in sorted_idxs]

        individ.structure = target_tokens
        individ.add_substructure(other_tokens)
        individ.fixator = tmp_fixator

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

