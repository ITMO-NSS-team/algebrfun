import numpy as np

from sklearn.linear_model import LinearRegression
 
from .base import GeneticOperatorIndivid
from .base import GeneticOperatorPopulation
from .base import apply_decorator

class LRIndivid(GeneticOperatorIndivid):
    def __init__(self, params: dict = None):
        super().__init__(params)

    def _prepare_data(self, individ):
        target = []
        features = []
        grid = self.params['grid']

        for term in individ.structure:
            value_on_grid = term.value(grid)
            if term.mandatory == 0:
                features.append(value_on_grid)
            else:
                target.append(value_on_grid)
        
        return -np.sum(target, axis=0), np.array(features)

    def _set_amplitudes(self, individ, coef):
        coefs = list(coef)
        idx = 0

        for term in individ.structure:
            if term.mandatory != 0:
                continue
            new_aplitude = term.expression_token.param(name='Amplitude') * coefs[idx]
            idx += 1
            if np.abs(new_aplitude[0]) < 1:
                continue
            term.expression_token.set_param(new_aplitude, name='Amplitude')

    @apply_decorator
    def apply(self, individ, *args, **kwargs):
        target, features = self._prepare_data(individ=individ)

        model = LinearRegression(fit_intercept=False)
        model.fit(features.T, target)

        self._set_amplitudes(individ, model.coef_)


class DecimationPopulation(GeneticOperatorPopulation):
    def __init__(self, params: dict = None):
        super().__init__(params)

    def apply(self, population, *args, **kwargs):
        individs = population.structure
        for individ in individs:
            individ.apply_operator("FilterIndivid")
            tokens = list(filter(lambda elem: not elem.mandatory, individ.structure))
            tokens = sorted(tokens, key=lambda elem: elem.fitness, reverse=True)
            for token in tokens:
                temp_individ = individ.copy()
                temp_individ.structure.remove(token)
                if len(temp_individ.structure) <= 2:
                    continue
                temp_individ.apply_operator("LRIndivid")
                temp_individ.apply_operator("VarFitnessIndivid")

                if temp_individ.fitness < individ.fitness:
                    individ.structure.remove(token)
                    individ.apply_operator("LRIndivid")
                    individ.apply_operator("VarFitnessIndivid")

