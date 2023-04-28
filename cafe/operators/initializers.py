import numpy as np

from cafe.operators.base import GeneticOperatorIndivid
from cafe.operators.base import GeneticOperatorPopulation
from cafe.operators.base import apply_decorator


class InitIndivid(GeneticOperatorIndivid):
    def __init__(self, params=None):
        super().__init__(params=params)


    @apply_decorator
    def apply(self, individ, *args, **kwargs) -> None:
        terms = list(filter(lambda term: term.mandatory == 0, self.params['terms']))
        tokens = self.params['tokens']

        sub = self.params['terms'].copy()

        for trm in sub:
            if trm.mandatory:
                continue
            current_token = np.random.choice(tokens).copy()
            current_token._select_params()
            trm.expression_token = current_token

        number_of_opt_terms = len(terms) * len(tokens) // 2
        # count_of_added = 0

        for count_of_added in range(number_of_opt_terms):
            current_term = np.random.choice(terms).copy()
            current_token = np.random.choice(tokens).copy()

            # current_token._select_params(self.params['shape'])
            current_token._select_params()

            current_term.expression_token = current_token
            if current_term not in individ.structure:
                sub.append(current_term)
                # count_of_added += 1
        individ.add_substructure(sub)


class InitPopulation(GeneticOperatorPopulation):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('population_size', 'individ')

    def apply(self, population, *args, **kwargs):
        population.structure = []
        for _ in range(self.params['population_size']):
            new_individ = self.params['individ'].copy()
            new_individ.apply_operator('InitIndivid')
            population.structure.append(new_individ)
        return population