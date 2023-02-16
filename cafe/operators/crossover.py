import random
from itertools import groupby
import numpy as np

from .base import GeneticOperatorIndivid
from .base import GeneticOperatorPopulation
from .base import apply_decorator


class CrossoverIndivid(GeneticOperatorIndivid):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('cross_intensive', 'increase_prob')

    @apply_decorator
    def apply(self, individ, *args, **kwargs): #TODO: не добавлять заведомо присутствующие токены
        cross_intensive = self.params['cross_intensive']
        increase_prob = self.params['increase_prob']


        ind1 = individ
        ind2 = kwargs['other_individ']

        tokens1 = dict((k, list(i)) for k, i in groupby(ind1.structure, key=lambda elem: elem.name_))
        tokens2 = dict((k, list(i)) for k, i in groupby(ind2.structure, key=lambda elem: elem.name_))

        for key in tokens1.keys():
            expression1 = tokens1.get(key)
            expression2 = tokens2.get(key)

            if expression1[0].mandatory:
                continue

            cross_intensive = np.min([cross_intensive, len(tokens1), len(tokens2)])
            add_tokens1, add_tokens2 = tuple(map(lambda tokens: np.random.choice(tokens, size=cross_intensive,
                                                                             replace=False), (expression1, expression2)))

            if np.random.uniform() < increase_prob:
                for token in add_tokens1:
                    token_copy = token.copy()
                    ind2.add_substructure([token_copy])
                for token in add_tokens2:
                    token_copy = token.copy()
                    ind1.add_substructure([token_copy])
            else:
                for token1, token2 in np.transpose([add_tokens1, add_tokens2]):
                    tmp_token1, tmp_token2 = token1.copy(), token2.copy()
                    ind1.set_substructure(tmp_token2, ind1.structure.index(token1))
                    ind2.set_substructure(tmp_token1, ind2.structure.index(token2))


class CrossoverPopulation(GeneticOperatorPopulation):
    def __init__(self, params):
        super().__init__(params=params)
        # self._check_params('crossover_size')

    def apply(self, population, *args, **kwargs):
        selected_population = list(filter(lambda individ: individ.selected, population.structure))
        crossover_size = self.params['crossover_size']
        if crossover_size is None or crossover_size > len(selected_population)//2:
            crossover_size = len(selected_population)//2
        selected_individs = np.random.choice(selected_population, replace=False, size=(crossover_size, 2))

        for individ1, individ2 in selected_individs:
            n_ind_1 = individ1.copy()
            n_ind_2 = individ2.copy()
            n_ind_1.elitism = False
            n_ind_2.elitism = False
            population.structure.extend([n_ind_1, n_ind_2])
            n_ind_1.apply_operator('CrossoverIndivid', other_individ=n_ind_2)  # Параметры мутации заключены в операторе мутации с которым
        return population