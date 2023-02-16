import numpy as np

from scipy.optimize import differential_evolution
from scipy.optimize import minimize

from .base import GeneticOperatorIndivid
from .base import GeneticOperatorPopulation
from .base import apply_decorator
from cafe.settings import FrequencyProcessor4TimeSeries as fp

class TokenParametersOptimizerIndivid(GeneticOperatorIndivid):
    """
    """

    def __init__(self, params=None):
        if params is None:
            params = {}
        add_params = {
            'optimizer': 'DE',
            'popsize': 7,
            'eps': 0.005
        }
        for key, value in add_params.items():
            if key not in params.keys():
                params[key] = value
        super().__init__(params=params)
        # self._check_params('grid', 'optimizer', 'optimize_id', 'popsize', 'eps')

    def _choice_periodic_tokens(self, individ):
        optimize_id = 1

        choiced_tokens = list(filter(lambda token: token.expression_token.optimize_id == optimize_id, individ.structure))
        return choiced_tokens

    def preprocess_periodic_tokens(self, individ, tokens):
        grid = self.params['grid']
        shp = self.params['shape']
        eps = self.params['eps']

        for token_id in range(len(tokens)):
            target = -tokens[token_id].value(grid)
            target -= target.min()

            freq = fp.choice_freq_for_summand(grid, target-target.mean(), shp, number_selecting=5, number_selected=5, token_type='seasonal')

            if freq is None:
                individ.structure.remove(tokens[token_id])
            else:
                params = tokens[token_id].expression_token.params

                index = tokens[token_id].expression_token.get_key_use_params_description(descriptor_name='name',
                                                    descriptor_value='Frequency')
                for i in range(params.shape[0]):
                    params[i][index] = freq[0][i]
                tokens[token_id].expression_token.set_descriptor(index, 'bounds', ((freq[0][0] * (1 - eps), freq[0][0] * (1 + eps))))
        # return tokens

    @staticmethod
    def _fitness_wrapper(params, *args):
        individ, grid, shp = args
        i = 0

        for term in individ.structure:
            if term.mandatory:
                continue
            token = term.expression_token
            number_of_params = token._number_params * shp[0]
            k = i + number_of_params
            token.params = params[i:k].reshape(shp[0], token._number_params)
            i = k
        
        individ.fitness = None
        individ.apply_operator('VarFitnessIndivid')

        return individ.fitness

    def _optimize_tokens_params(self, individ):
        grid = self.params['grid']
        shp = self.params['shape']

        choice_terms = list(filter(lambda term: not term.mandatory, individ.structure))
        if self.params['optimizer'] == "DE":
            bounds = []
            for term in choice_terms:
                for param in term.expression_token.params_description:
                    if term.expression_token.params_description[param]['name'] == 'Amplitude':
                        bounds.append(term.expression_token.params_description[param]['bounds'])
                        continue
                    for i in range(shp[0]):
                        bounds.append(term.expression_token.params_description[param]['bounds']) 
            
            res = differential_evolution(self._fitness_wrapper, bounds, args=(individ, grid, shp), popsize=self.params['popsize'])
        else:
            x0 = []
            for term in choice_terms:
                for param in term.expression_token.params:
                    x0.extend(param)
            x0 = np.array(x0)
            res = minimize(self._fitness_wrapper, x0, args=(individ, grid, shp))
        
        i = 0
        result_params = res.x
        for term in individ.structure:
            if term.mandatory:
                continue
            token = term.expression_token
            number_of_params = token._number_params * shp[0]
            k = i + number_of_params
            token.params = result_params[i:k].reshape(shp[0], token._number_params)
            i = k

    @apply_decorator
    def apply(self, individ, *args, **kwargs):
        periodic_tokens = self._choice_periodic_tokens(individ)
        if len(periodic_tokens) != 0:
            self.preprocess_periodic_tokens(individ, periodic_tokens)
        self._optimize_tokens_params(individ)

class TokenParametersOptimizerPopulation(GeneticOperatorPopulation):
    """
    """

    def __init__(self, params: dict = None):
        super().__init__(params=params)

    def apply(self, population, *args, **kwargs) -> None:
        for individ in population.structure:
            individ.apply_operator('TokenParametersOptimizerIndivid')