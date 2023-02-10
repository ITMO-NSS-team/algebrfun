import numpy as np

from sklearn.linear_model import LinearRegression
 
from .base import GeneticOperatorIndivid
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

        model = LinearRegression(fit_intercept=True)
        model.fit(features.T, target)

        self._set_amplitudes(individ, model.coef_)

