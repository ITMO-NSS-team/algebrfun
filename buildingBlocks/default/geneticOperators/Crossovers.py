"""Реализация операторов кроссовера для индивида и популяции, работают inplace"""

from audioop import cross
from multiprocessing import current_process

# from buildingBlocks.baseline.GeneticOperators import GeneticOperatorIndivid, GeneticOperatorPopulation
from buildingBlocks.baseline.BasicEvolutionaryEntities import GeneticOperatorIndivid, GeneticOperatorPopulation
import numpy as np

from buildingBlocks.default.geneticOperators.supplementary.Other import check_operators_from_kwargs, apply_decorator


class CrossoverIndivid(GeneticOperatorIndivid):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('cross_intensive', 'increase_prob')

    @apply_decorator
    def apply(self, individ, *args, **kwargs): #TODO: не добавлять заведомо присутствующие токены
        cross_intensive = self.params['cross_intensive']
        increase_prob = self.params['increase_prob']
        # ind1 = individs[0].copy() изменяются сами
        # ind2 = individs[1].copy()

        ind1 = individ
        ind2 = kwargs['other_individ']

        # ind1.kind += '->crossover'
        # ind2.kind += '->crossover'

        tokens1 = ind1.structure
        tokens2 = ind2.structure

        # inds change tokens or add them to each other depending on increase prob
        # tokens remain fixed!
        cross_intensive = np.min([cross_intensive, len(tokens1), len(tokens2)])
        add_tokens1, add_tokens2 = tuple(map(lambda tokens: np.random.choice(tokens, size=cross_intensive,
                                                                             replace=False),
                                         (tokens1, tokens2)))

        # add_idxs1 = np.random.choice(len(ind1.structure), size=cross_intensive, replace=False)
        # add_idxs2 = np.random.choice(len(ind2.structure), size=cross_intensive, replace=False)
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
        self._check_params('crossover_size')

    def apply(self, population, *args, **kwargs):
        selected_population = list(filter(lambda individ: individ.selected, population.structure))
        crossover_size = self.params['crossover_size']
        if crossover_size is None or crossover_size > len(selected_population)//2:
            crossover_size = len(selected_population)//2
        # else:
        #     assert crossover_size <= len(selected_population), "Crossover size in pairs" \
        #                                                        " must be less than population size"
        selected_individs = np.random.choice(selected_population, replace=False, size=(crossover_size, 2))

        for individ1, individ2 in selected_individs:
            # for individ in individ1, individ2:
            #     if individ.elitism:
            #         individ.elitism = False
            #         new_individ = individ.copy()
            #         new_individ.selected = False
            #         population.structure.append(new_individ)
            n_ind_1 = individ1.copy()
            n_ind_2 = individ2.copy()
            population.structure.extend([n_ind_1, n_ind_2])
            n_ind_1.apply_operator('CrossoverIndivid', other_individ=n_ind_2)  # Параметры мутации заключены в операторе мутации с которым
        return population
