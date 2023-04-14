import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution
from scipy.optimize import minimize, minimize_scalar

from .base import GeneticOperatorIndivid
from .base import GeneticOperatorPopulation
from .base import apply_decorator
from cafe.settings import FrequencyProcessor4TimeSeries as fp

def convert(params, number_of_params, ndim):
    k = 0
    res = []
    if number_of_params == 1:
        return params
    for d in range(number_of_params):
        if not d:
            res.append([params[k] for _ in range(ndim)])
            k += 1
        else:
            res.append([params[k+i] for i in range(ndim)])
            k += ndim
    
    return np.array(res).T

class TokenParametersOptimizerIndivid(GeneticOperatorIndivid):
    """
    """

    def __init__(self, params=None):
        if params is None:
            params = {}
        add_params = {
            'optimizer': 'DE',
            'popsize': 7,
            'eps': 0.1
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
    
    def _choice_trend_tokens(self, individ):
        optimize_id = 2

        choiced_tokens = list(filter(lambda token: token.expression_token.optimize_id == optimize_id, individ.structure))
        return choiced_tokens

    def preprocess_tokens(self, individ, tokens, token_type):
        grid = self.params['grid']
        shp = self.params['shape']
        eps = self.params['eps']

        for token_id in range(len(tokens)):
            # target = -individ.value(grid)
            # target = -tokens[token_id].data
            # target -= target.min()

            # target = self.params['target']
            target = tokens[token_id].data
            # print(individ.formula())

            # plt.plot(grid[0], target)
            # plt.show()

            freq, steps = fp.choice_freq_for_summand(grid, target, shp, number_selecting=5, number_selected=5, token_type=token_type)
            # print(freq[0][0] * (1 - steps[0]))
            # print(f"freqs: {steps}")

            if freq is None:
                individ.structure.remove(tokens[token_id])
            else:
                params = tokens[token_id].expression_token.params

                index = tokens[token_id].expression_token.get_key_use_params_description(descriptor_name='name',
                                                    descriptor_value='Frequency')
                for i in range(params.shape[0]):
                    params[i][index] = freq[0][i]
                tokens[token_id].expression_token.set_descriptor(index, 'bounds', ((freq[0][0] * (1 - np.abs(steps[0])), freq[0][0] * (1 + np.abs(steps[0])))))
        # return tokens

    @staticmethod
    def _fitness_wrapper_prep(param, *args):
        individ, count_ax, token, var_range = args
        token_params = []

        for i in range(count_ax):
            # axs_params = []
            params = var_range[i][int(param[i])]
            for j, pm in enumerate(params):
                if token.name_ == "target":
                    k = j
                else:
                    k = j + 1
                try:
                    token.params[i][k] = pm
                except:
                    temp_params = list(token.params)
                    temp_params.append(np.copy(temp_params[i - 1]))
                    token.params.params = np.array(temp_params)
                    # token.params.append(np.copy(token.params[i - 1]))
                    token.params[i][k] = pm
        # token.params = np.array(token_params)
        
        individ.fitness = None
        try:
            individ.apply_operator('VarFitnessIndivid')
        except:
            print(token.params)

        return individ.fitness
    
    def preprocess_tokens_(self, individ, tokens):
        grid = self.params['grid']
        eps = self.params['eps']

        for token_id, token in enumerate(tokens):
            x0 = np.array([0 for i in range(grid.shape[0])])
            caf_token = token.expression_token
            if caf_token.name_ == "target":
                continue
            try:
                var_range = caf_token.variable_params
            except:
                continue

            res = minimize(self._fitness_wrapper_prep, x0, args=(individ, grid.shape[0], caf_token, var_range))

            for i in range(grid.shape[0]):
                params = var_range[i][int(res.x[i])]
                for k, param in enumerate(params):
                    if caf_token.name_ == "target":
                        index = k
                    else:
                        index = k + 1
                    tokens[token_id].expression_token.set_descriptor(index, 'bounds', ((param - eps, param + eps)))
                    try:
                        tokens[token_id].expression_token.params[i][k] = param
                    except:
                        temp_params = list(tokens[token_id].expression_token.params)
                        temp_params.append(np.copy(temp_params[i - 1]))
                        tokens[token_id].expression_token.params = np.array(temp_params)
                        # tokens[token_id].params.append(np.copy(token.params[i - 1]))
                        tokens[token_id].expression_token.params[i][k] = param

            # tokens[token_id].expression_token.params = var_range[int(res.x)]
                 
    @staticmethod
    def _fitness_wrapper(params, *args):
        individ, grid, shp, cur_term = args
        i = 0

        if not cur_term:
            for term in individ.structure:
                if term.mandatory:
                    continue
                token = term.expression_token
                number_of_params = 1 + (token._number_params - 1) * grid.shape[0]
                k = i + number_of_params
                ch_params = params[i:k]
                try:
                    # token.params = params[i:k].reshape(grid.shape[0], token._number_params)
                    token.params = convert(ch_params, token._number_params, grid.shape[0])
                except Exception as e:
                    print(f"wrapper: {e}\n {params[i:k]}")
                    raise ValueError("Sizes is bad")
                i = k
        else:
            token = cur_term.expression_token
            number_of_params = 1 + (token._number_params - 1) * grid.shape[0]
            k = i + number_of_params
            ch_params = params[i:k]
            try:
                # token.params = params[i:k].reshape(grid.shape[0], token._number_params)
                token.params = convert(ch_params, token._number_params, grid.shape[0])
            except Exception as e:
                print(f"wrapper: {e}\n {params[i:k]}")
                raise ValueError("Sizes is bad")
            i = k
        
        individ.fitness = None
        individ.apply_operator('VarFitnessIndivid')

        return individ.fitness

    def _optimize_tokens_params(self, individ, token=None):
        grid = self.params['grid']
        shp = self.params['shape']
        eps = self.params['eps']

        if not token:
            choice_terms = list(filter(lambda term: not term.mandatory, individ.structure))
        else:
            choice_terms = [token]

        # self.preprocess_tokens_(individ, choice_terms)

        if self.params['optimizer'] == "DE":
            bounds = []
            for term in choice_terms:
                excpression_name = term.expression_token.name_
                if term.expression_token._number_params > 1:
                    term.expression_token.select_best_params(individ)
                for param in term.expression_token.params_description:
                    if term.expression_token.params_description[param]['name'] == 'Amplitude':
                        bounds.append(term.expression_token.params_description[param]['bounds'])
                        continue
                    for i in range(grid.shape[0]):
                        try:
                            bounds.append((term.expression_token.params[i][param] - eps, term.expression_token.params[i][param] + eps))
                        except:
                            bounds.append(term.expression_token.params_description[param]['bounds'])
                        # bounds.append(term.expression_token.params_description[param]['bounds']) 
            try:
                # print("bounds:", len(bounds))
                res = differential_evolution(self._fitness_wrapper, bounds, args=(individ, grid, shp, token), popsize=self.params['popsize'])
            except Exception as e:
                print(f"optimizer: {e}\n{bounds}")
                exit()
        else:
            x0 = []
            for term in choice_terms:
                for param in term.expression_token.params:
                    x0.extend(param)
            x0 = np.array(x0)
            res = minimize(self._fitness_wrapper, x0, args=(individ, grid, shp))
        
        i = 0
        result_params = res.x

        if token:
            expres = term.expression_token
            number_of_params = 1 + (expres._number_params - 1) * grid.shape[0]
            k = i + number_of_params
            # token.params = result_params[i:k].reshape(grid.shape[0], token._number_params)
            expres.params = convert(result_params[i:k], expres._number_params, grid.shape[0])
            i = k
            return
        for term in individ.structure:
            if term.mandatory:
                continue
            token = term.expression_token
            number_of_params = 1 + (token._number_params - 1) * grid.shape[0]
            k = i + number_of_params
            # token.params = result_params[i:k].reshape(grid.shape[0], token._number_params)
            token.params = convert(result_params[i:k], token._number_params, grid.shape[0])
            i = k

    @apply_decorator
    def apply(self, individ, *args, **kwargs):
        # periodic_tokens = self._choice_periodic_tokens(individ)
        # if len(periodic_tokens) != 0:
        #     self.preprocess_tokens(individ, periodic_tokens, 'seasonal')
            # print([tkn.expression_token.params_description for tkn in periodic_tokens])
        
        # trend_tokens = self._choice_trend_tokens(individ)
        # if len(trend_tokens) != 0:
        #     self.preprocess_tokens_(individ, trend_tokens)
            
        # self._optimize_tokens_params(individ)

        for term in individ.structure:
            if term.mandatory:
                continue
            if term.expression_token.name_ == 'ComplexToken':
                tkns = term.get_complex()
                for i, tkn in enumerate(tkns):
                    i_term = term.copy()
                    i_term.expression_token = tkns[i]
                    self._optimize_tokens_params(individ, i_term)
            else:
                self._optimize_tokens_params(individ, term)

class TokenParametersOptimizerPopulation(GeneticOperatorPopulation):
    """
    """

    def __init__(self, params: dict = None):
        super().__init__(params=params)

    def apply(self, population, *args, **kwargs) -> None:
        for individ in population.structure:
            if individ.elitism:
                continue
            individ.apply_operator('TokenParametersOptimizerIndivid')
