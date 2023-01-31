import random
import numpy as np

from base import GeneticOperatorIndivid
from base import GeneticOperatorPopulation
from base import apply_decorator

class MutationIndivid(GeneticOperatorIndivid):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('mut_intensive', 'increase_prob', 'tokens')

    @apply_decorator
    def apply(self, individ, *args, **kwargs) -> None:
        # individ.kind += '->mutation'
        tokens = list(filter(lambda token: token not in individ.structure, self.params['tokens']))
        mandatory_tokens = list(filter(lambda token: token.mandatory != 0, tokens))
        non_mandatory_tokens = list(filter(lambda token: token.mandatory == 0, tokens))

        if not tokens:
            return

        mut_intensive = self.params['mut_intensive']
        if mandatory_tokens:
            individ.add_substructure([token.clean_copy() for token in mandatory_tokens])

        add_tokens = []
        if mut_intensive > len(non_mandatory_tokens):
            add_tokens.extend(non_mandatory_tokens)
        else:
            for _ in range(mut_intensive):
                while True:
                    new_token = np.random.choice(non_mandatory_tokens).clean_copy()
                    if new_token not in add_tokens:
                        add_tokens.append(new_token)
                        break
        random.shuffle(add_tokens)

        if individ.max_tokens > len(individ.structure) and np.random.uniform() <= self.params['increase_prob']:
            individ.add_substructure(add_tokens)
        elif individ.structure:
            individ.apply_operator('TokenFitnessIndivid')
            idxs_to_choice = list(filter(lambda idx: individ.structure[idx].mandatory == 0, # and individ.structure no is CImp
                                         range(len(individ.structure))))
            if not idxs_to_choice:
                return
            try:
                test_prob = np.array(list(map(lambda idx: individ.structure[idx].fitness, idxs_to_choice)))
                probabilities = np.array(list(map(lambda idx: individ.structure[idx].fitness, idxs_to_choice)))
                probabilities /= probabilities.sum()
                for idx in np.random.choice(idxs_to_choice,
                                            size=min(len(idxs_to_choice), len(add_tokens)),
                                            replace=False,
                                            p=probabilities):
                    individ.set_substructure(add_tokens.pop(), idx)
            except:
                individ.apply_operator('TokenFitnessIndivid')

            if add_tokens:
                individ.add_substructure(add_tokens)

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