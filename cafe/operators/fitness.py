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
        # constants = get_full_constant()
        # # ftnss = constants['all_fitness']
        # if individ.type_ == "DEquation":
        #     individ.fitness = np.linalg.norm(individ.value(self.params['grid']))
        #     # ftnss['de'].append(individ.fitness)
        #     # set_constants(all_fitness=ftnss)
        #     return
        # b_individ = constants['best_individ'].copy()
        # b_individ.set_CAF(individ)
        # other_individs = constants['all_structures']
        # other_individ = np.random.choice(other_individs)
        # other_individ.set_CAF(individ)
        # for dtoken in b_individ:
        #     other_individ.set_CAF(dtoken.params[0])
        # fts_0 = np.linalg.norm(b_individ.value(self.params['grid']))
        # fts_1 = np.linalg.norm(other_individ.value(self.params['grid']))

        # if fts_1 < fts_0:
        #     set_constants(best_individ=other_individ)
        #     fts_0 = fts_1

        # ftnss['CAF'].append(individ.fitness)
        # set_constants(all_fitness=ftnss)

        individ.fitness = None


class TokenFitnessIndivid(GeneticOperatorIndivid):
    def __init__(self, params=None):
        super().__init__(params=params)
        self._check_params()  

    @staticmethod
    def _label_tokens(individ):
        other_tokens = list(filter(lambda token: token.mandatory == 0 and token.fixator['self'], individ.structure))
        if individ.type_ == "DEquation":
            target_tokens = list(filter(lambda token: token.mandatory != 0, individ.structure))
            assert len(target_tokens) == 1, 'Individ must have only one target token'
            other_tokens = list(filter(lambda token: token.mandatory == 0, individ.structure))
        if not other_tokens:
            return

        tmp_individ = individ.clean_copy()
        if individ.type_ == "DEquation":
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

        if individ.type_ == "DEquation":
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
            individ.apply_operator('VarFitnessIndivid', args[0])
        return population

