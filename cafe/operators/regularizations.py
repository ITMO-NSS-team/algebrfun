import numpy as np

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import normalize, scale

 
from .base import GeneticOperatorIndivid
from .base import GeneticOperatorPopulation
from .base import apply_decorator

class LRIndivid(GeneticOperatorIndivid):
    def __init__(self, params: dict = None):
        super().__init__(params)

    def lasso(self, individ):
        target, features = self._prepare_data_for_lasso(individ=individ)

        features -= np.mean(features, axis=1, keepdims=True)
        # target -= np.mean(target, keepdims=True)

        # features, norms = normalize(features, norm="max", axis=1, return_norm=True)

        X = features.T

        model = Lasso(self.params['regularisation_coef'], fit_intercept=False)
        model.fit(X, target)
        
        coefs = model.coef_
        new_chromo = []
        idxs = 0
        
        for term in individ.structure:
            if term.mandatory:
                continue
            if coefs[idxs] != 0:
                new_chromo.append(term)
            
            idxs += 1

        individ.structure = new_chromo

    
    def _prepare_data_for_lasso(self, individ):
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

    def _prepare_data(self, individ):
        target = []
        features = []
        grid = self.params['grid']

        for term in individ.structure:
            if term.expression_token.name_ == "ComplexToken":
                for token in term.get_complex():
                    cur_term = term.copy()
                    cur_term.expression_token = token
                    value_on_grid = cur_term.value(grid)
                    if term.mandatory == 0:
                        features.append(value_on_grid)
                    else:
                        target.append(value_on_grid)
            else:
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
            if term.expression_token.name_ == "ComplexToken":
                for token in term.get_complex():
                    new_aplitude = token.param(name='Amplitude') * coefs[idx]
                    idx += 1
                    if np.abs(new_aplitude[0]) < 1:
                        continue
                    token.set_param(new_aplitude, name='Amplitude')
            else:
                new_aplitude = term.expression_token.param(name='Amplitude') * coefs[idx]
                idx += 1
                # if np.abs(new_aplitude[0]) < 1:
                #     continue
                term.expression_token.set_param(new_aplitude, name='Amplitude')

    @apply_decorator
    def apply(self, individ, *args, **kwargs):
        # self.lasso(individ=individ)

        target, features = self._prepare_data(individ=individ)

        model = LinearRegression(fit_intercept=False)
        # if len(features) == 0 or len(target) == 0:
        #     return 
        try:
            model.fit(features.T, target)
        except:
            return

        self._set_amplitudes(individ, model.coef_)

class ClearComplexTokens(GeneticOperatorPopulation):
    def __init__(self, params: dict = None):
        super().__init__(params)

    def apply(self, population, *args, **kwargs):
        individs = population.structure
        for individ in individs:
            if individ.elitism:
                continue
            tokens = list(filter(lambda elem: not elem.mandatory and elem.expression_token.name_ == "ComplexToken", individ.structure))
            tokens = sorted(tokens, key=lambda elem: elem.fitness, reverse=True)

            for token in tokens:
                temp_individ = individ.copy()
                temp_token = token.expression_token.copy()
                del_token = None
                fts = temp_individ.fitness.copy()

                for tkn in temp_token.tokens:
                    token.expression_token.tokens.remove(tkn)
                    if len(token.expression_token.tokens) == 0:
                        individ.structure.remove(token)
                        individ.apply_operator("LRIndivid")
                        individ.apply_operator("VarFitnessIndivid")
                        individ.structure.append(token)
                    else:
                        individ.apply_operator("LRIndivid")
                        individ.apply_operator("VarFitnessIndivid")

                    if individ.fitness < fts:
                        del_token = tkn
                        fts = individ.fitness.copy()
                        
                    token.expression_token.tokens.append(tkn)
                    # if individ.fitness > temp_individ.fitness:
                    #     individ.structure.append(token)
                    #     token.expression_token.tokens = temp_token.tokens
                try:
                    individ.structure.remove(token)
                except:
                    pass
                individ.apply_operator("VarFitnessIndivid")
                if individ.fitness > fts:
                    individ.structure.append(token)
                    if del_token:
                        token.expression_token.tokens.remove(del_token)
                        if len(token.expression_token.tokens) == 0:
                            individ.structure.remove(token)
                
                individ.apply_operator("VarFitnessIndivid")

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

