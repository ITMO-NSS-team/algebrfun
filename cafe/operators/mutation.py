from itertools import groupby
import random
import numpy as np

from .base import GeneticOperatorIndivid
from .base import GeneticOperatorPopulation
from .base import apply_decorator

class MutationIndivid(GeneticOperatorIndivid):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('mut_intensive', 'increase_prob', 'tokens')

    @apply_decorator
    def apply(self, individ, *args, **kwargs) -> None:
        mut_intensive = self.params['mut_intensive']

        # group_by_term = dict((k, list(i)) for k, i in groupby(individ.structure, key=lambda elem: elem.name_))
        group_by_term = dict((k, list(filter(lambda elem: elem.name_ == k, individ.structure))) for k in np.unique(list(map(lambda elem: elem.name_, individ.structure))))

        for name_term in group_by_term.keys():
            full_term = group_by_term.get(name_term)
            if full_term[0].mandatory:
                continue
            tmp_temp = full_term[0].copy()
            expression_structure = [term.expression_token for term in full_term]
            tokens = list(filter(lambda token: token not in expression_structure, self.params['tokens']))
            for i, token in enumerate(tokens):
                token._select_params()
                tmp_temp.expression_token = token
                tokens[i] = tmp_temp.copy()

            if not tokens:
                continue

            add_tokens = []

            if mut_intensive > len(expression_structure):
                add_tokens.extend(tokens)
            else:
                new_tokens = np.random.choice(tokens, mut_intensive, replace=False)
                for new_token in new_tokens:
                    add_tokens.append(new_token.copy())
            
            random.shuffle(add_tokens)

            if individ.max_tokens > len(expression_structure) and np.random.uniform() <= self.params['increase_prob']:
                individ.add_substructure(add_tokens)
            else:
                probabilities = np.array(list(map(lambda idx: full_term[idx].fitness, range(len(full_term)))))
                probabilities /= probabilities.sum()
                for idx in np.random.choice(np.arange(len(full_term)),
                                        size=min(len(full_term), len(add_tokens)),
                                        replace=False,
                                        p=probabilities):
                    add_token = add_tokens.pop()
                    full_term[idx].expression_token = add_token.expression_token
                if add_tokens:
                    individ.add_structure(add_tokens)   

class MutationPopulation(GeneticOperatorPopulation):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('mutation_size')

    def apply(self, population, *args, **kwargs):
        selected_population = list(filter(lambda individ: individ.selected, population.structure))
        mutation_size = self.params['mutation_size']
        if mutation_size is None or mutation_size > len(selected_population):
            selected_individs = selected_population
        else:
            # assert mutation_size <= len(selected_population), "Mutations size must be less than population size"
            selected_individs = np.random.choice(selected_population, replace=False, size=mutation_size)

        for individ in selected_individs:
            if individ.elitism:
                individ.elitism = False
                new_individ = individ.copy()
                new_individ.selected = False
                population.structure.append(new_individ)
            individ.apply_operator('MutationIndivid')
                # individ.apply_operator('ImpComplexMutationIndivid')
            # individ.apply_operator('ProductTokenMutationIndivid')
        return population