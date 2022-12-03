"""
Contains Genetic Operators responsible for length of individ's chromosomes.
The chromosome can be either shortened or expanded.

Classes
----------
LassoIndivid
DEOptIndivid
DelDuplicateTokensIndivid
CheckMandatoryTokensIndivid

LassoPopulation
RegularisationPopulation
"""
import random
from copy import copy
from typing import List

from scipy.optimize import differential_evolution
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import normalize, scale
# from buildingBlocks.baseline.GeneticOperators import GeneticOperatorIndivid, GeneticOperatorPopulation
from buildingBlocks.baseline.BasicEvolutionaryEntities import GeneticOperatorIndivid, GeneticOperatorPopulation
from buildingBlocks.baseline.BasicEvolutionaryEntities import ComplexToken
import numpy as np
from multiprocessing import current_process

from buildingBlocks.default.geneticOperators.supplementary.Other import check_or_create_fixator_item, \
    check_operators_from_kwargs, apply_decorator
import buildingBlocks.baseline.BasicEvolutionaryEntities as Bee


class LassoIndivid(GeneticOperatorIndivid): #TODO не удалять токены а повысить шанс их мутации/кроссовера/ухода из хромосомы
    def __init__(self, params):
        super().__init__(params=params)

    def _preprocessing_data(self, individ, normalize_=True):
        """
        Нормирует все данные, выбирает наиболее значимую фичу по норме и представляет ее в качестве таргета
        Args:
            individ:

        Returns:

        """
        chromo = individ.get_structure()
        features = np.transpose(np.array(list(map(lambda token: token.value(self.params['grid']), chromo))))
        global_features = np.transpose(np.array(list(map(lambda token: token.value(self.params['grid']), individ.structure))))
        if normalize_:
            # features -= np.mean(features, axis=0, keepdims=True)
            # features, norms = normalize(features, norm='l2', axis=0, return_norm=True)

            mandatory_idxs = [idx for idx in range(len(chromo)) if chromo[idx].mandatory != 0]
            if len(mandatory_idxs) == 1:
                target_idx = mandatory_idxs[0]
            elif len(mandatory_idxs) == 2:
                cov = np.cov(features.T[mandatory_idxs])
                idx = np.argmax([cov[i][i] for i in range(2)])
                target_idx = mandatory_idxs[idx]
            else:
                if len(mandatory_idxs) == 0:
                    # target_idx = np.random.randint(0, len(chromo))
                    new_features = features.T
                    mandatory_idxs = [idx for idx in range(len(chromo))]
                else:
                    # target_idx = np.random.choice(mandatory_idxs)
                    new_features = features.T[mandatory_idxs]
                cov = np.abs(np.corrcoef(new_features))
                for i in range(len(cov)):
                    cov[i][i] = 0
                x, y = np.unravel_index(cov.argmax(), cov.shape)
                target_idx = mandatory_idxs[np.random.choice((x, y))]

                # target_idx = mandatory_idxs[np.argmax(cov.sum(axis=1))]
            # print('lasso target idx--->', target_idx, ' ', individ.structure[target_idx].name())
        else:
            # target_idx = np.argmax(np.var(features, axis=0))
            # target_idx = np.random.randint(features.shape[1])
            if features.T.shape[0] >= 3:
                mandatory_idxs = [idx for idx in range(len(individ.structure)) if individ.structure[idx].mandatory != 0]
                target_idx = mandatory_idxs[0]
            else:
                cov = np.cov(features.T)
                target_idx = np.argmax([cov[i][i] for i in range(2)])
            # print('LR target idx--->', target_idx, ' ', individ.structure[target_idx].name())
        target = -global_features[:, target_idx]
        idxs = [i for i in range(features.shape[1]) if chromo[i].params[1]._name != individ.structure[target_idx].params[1]._name]
        features = features[:, idxs]
        return features, target, target_idx

    def _regression_cascade(self, individ, lasso=True):
        print("REGRESSION CASCADE")
        chromo = individ.get_structure()
        features = np.array(list(map(lambda token: token.value(self.params['grid']), chromo)))
        features -= np.mean(features, axis=1, keepdims=True)
        big_features = np.array(list(map(lambda token: token.value(self.params['grid']), individ.structure)))
        big_features -= np.mean(big_features, axis=1, keepdims=True)
        features, norms = normalize(features, norm='max', axis=1, return_norm=True) #TOdo изменить на norm=max
        big_features, norms = normalize(big_features, norm='max', axis=1, return_norm=True)

        models = []
        for idx in range(len(big_features)):
            if individ.structure[idx].mandatory == 0:
                continue
            target = big_features[idx]
            idxs = [i for i in range(len(features)) if chromo[i].params[1]._name == individ.structure[idx].params[1]._name]
            X = features[[i for i in range(len(features)) if i not in idxs]].T
            if lasso:
                model = Lasso(self.params['regularisation_coef'])
            else:
                model = LinearRegression()
            model.fit(X, target)
            models.append((idx, model, model.score(X, target)))
        if len(models) == 0:
            print("pizda")
        print([model[2] for model in models])
        models.sort(key=lambda model: -model[2])
        return models[0]

    @staticmethod
    def _set_amplitudes_after_regression(individ, coefs, target_idx):  # , norms):
        print("SET AMPLITUDES>>>")
        chromo = individ.get_structure()
        l = len(chromo)
        new_chromo = []
        name_of_target = individ.structure[target_idx].params[1]._name
        for idx in range(l-1, -1, -1):
            if chromo[idx].params[1]._name == name_of_target:
                new_chromo.append(chromo.pop(idx))
        # chromo.pop(target_idx)
        for idx, token in enumerate(chromo):
            token.params[0].structure[0].set_param(token.params[0].structure[0].param(name='Amplitude') * coefs[idx], name='Amplitude')  # /norms[idx])
        new_chromo.extend(chromo)
        individ.set_structure(new_chromo)

    @staticmethod
    def _del_tokens_with_zero_coef(individ, coefs, target_idx):
        print("DEL TOKENS WITH>>>")
        print(individ.formula())
        name_of_target_term = individ.structure[target_idx].params[1]._name
        chromo = individ.get_structure()
        print('lasso target idx--->', target_idx, ' ', individ.structure[target_idx].name(), coefs)
        # new_chromo = [chromo.pop(target_idx)]
        new_chromo = []
        for idx, term_of_chromo in enumerate(chromo):
            if term_of_chromo.params[1]._name == name_of_target_term:
                new_chromo.append(chromo.pop(idx))

        # if (coefs == 0).all():
        #     individ.structure.reverse()
        #     individ.structure.extend(new_chromo)
        #     individ.structure.reverse()
        #     return
        for idx, coef in enumerate(coefs):
            if coef != 0 or chromo[idx].mandatory != 0: #or (individ.structure[idx].mandatory == 0
                                                                 #  and not individ.structure[idx].fix):
                new_chromo.append(chromo[idx])
        # idxs = [i for i in range(len(coefs))]
        # random.shuffle(idxs)
        # for idx in idxs:
        #     if coefs[idx] == 0 and len(new_chromo) < 2 and individ.structure[idx] not in new_chromo:
        #         new_chromo.append(individ.structure[idx])
        # individ.structure = new_chromo # ???
        print("chromo:", [elem.name() for elem in new_chromo])
        individ.set_structure(new_chromo)
        print("CHROMO END")

    def linear_regression(self, individ):
        print("LINEAR REGRESSION")
        if len(individ.structure) <= 1:
            return
        model = LinearRegression(fit_intercept=True)
        features, target, target_idx = self._preprocessing_data(individ, normalize_=False)
        model.fit(features, target)
        self._set_amplitudes_after_regression(individ, model.coef_, target_idx)

        try:
            individ.forms.append(type(self).__name__ + individ.formula() + '<---' + current_process().name)
        except:
            pass

    def lasso(self, individ, lasso=True):
        print("LASSO")
        if len(individ.structure) <= 2:
            return
        # model = Lasso(self.params['regularisation_coef'])
        # features, target, target_idx = self._preprocessing_data(individ)
        # model.fit(features, target)
        target_idx, model, _ = self._regression_cascade(individ, lasso=lasso)
        if lasso:
            self._del_tokens_with_zero_coef(individ, model.coef_, target_idx)

        try:
            individ.forms.append(type(self).__name__ + individ.formula() + '<---' + current_process().name)
        except:
            pass

    @apply_decorator
    def apply(self, individ, *args, **kwargs):
        print("APPLY", individ.formula())
        try:
            lasso = kwargs['use_lasso']
        except KeyError:
            lasso = True
        self.lasso(individ, lasso)
        self.linear_regression(individ)


class LassoIndivid1Target(GeneticOperatorIndivid):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('grid', 'regularisation_coef')

    @staticmethod
    def _find_target_idx(tokens):
        target_idxs = list(filter(lambda idx: tokens[idx].mandatory != 0,
                                  range(len(tokens))))
        assert len(target_idxs) == 1, 'Individ has no one or more than one target'
        return target_idxs[0]

    def _prepare_data(self, tokens):
        grid = self.params['grid']
        data = np.array(list(map(lambda token: token.value(grid), tokens)))
        # data -= np.mean(data, axis=1, keepdims=True)
        data -= data.min(axis=1, keepdims=True)
        data, norms = normalize(data, norm='max', axis=1, return_norm=True)
        return data, norms

    @staticmethod
    def _set_amplitudes(tokens, coefs, norms, target_idx):
        coefs = list(coefs)
        coefs.insert(target_idx, 1.)
        for idx, token in enumerate(tokens):
            token.set_param(token.param(name='Amplitude') / norms[idx] * coefs[idx], name='Amplitude')

    @staticmethod
    def _del_low_amplitude_tokens(individ):
        new_structure = []
        for idx, token in enumerate(individ.structure):
            if not token.fixator['self'] or token.mandatory != 0 or token.param(name='Amplitude') != 0:
                new_structure.append(token)
        individ.structure = new_structure

    def _lasso(self, individ):
        fixed_optimized_tokens_in_structure = list(filter(lambda token: token.fixator['self'],
                                                          individ.structure))
        if len(fixed_optimized_tokens_in_structure) <= 1:
            return
        target_idx = self._find_target_idx(fixed_optimized_tokens_in_structure)
        data, norms = self._prepare_data(fixed_optimized_tokens_in_structure)

        target = -data[target_idx]
        features = data[[idx for idx in range(data.shape[0]) if idx != target_idx]]

        model = Lasso(self.params['regularisation_coef'], fit_intercept=True)
        model.fit(features.T, target)

        self._set_amplitudes(fixed_optimized_tokens_in_structure, model.coef_, norms, target_idx)
        self._del_low_amplitude_tokens(individ)

    @apply_decorator
    def apply(self, individ, *args, **kwargs) -> None:
        self._lasso(individ)


class LRIndivid1Target(GeneticOperatorIndivid):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('grid')

    @staticmethod
    def _find_target_idx(tokens):
        target_idxs = list(filter(lambda idx: tokens[idx].mandatory != 0,
                                  range(len(tokens))))
        assert len(target_idxs) == 1, 'Individ has no one or more than one target'
        return target_idxs[0]

    def _prepare_data(self, tokens):
        grid = self.params['grid']
        data = np.array(list(map(lambda token: token.value(grid), tokens)))
        return data

    @staticmethod
    def _set_amplitudes(individ, tokens, coefs, target_idx):
        new_structure = []
        coefs = list(coefs)
        coefs.insert(target_idx, 1.)
        for idx, token in enumerate(tokens):
            new_amplitude = token.param(name='Amplitude') * coefs[idx]
            if np.abs(new_amplitude[0]) < 2:
                continue
            token.set_param(new_amplitude, name='Amplitude')
            new_structure.append(token)

        # individ.strucutre = new_structure

    def _lr(self, individ):
        fixed_optimized_tokens_in_structure = list(filter(lambda token: token.fixator['self'],
                                                          individ.structure))
        if len(fixed_optimized_tokens_in_structure) <= 1:
            return
        target_idx = self._find_target_idx(fixed_optimized_tokens_in_structure)
        data = self._prepare_data(fixed_optimized_tokens_in_structure)

        target = -data[target_idx]
        features = data[[idx for idx in range(data.shape[0]) if idx != target_idx]]

        # print("features & target", features, target)

        model = LinearRegression(fit_intercept=True)
        model.fit(features.T, target)

        self._set_amplitudes(individ, fixed_optimized_tokens_in_structure, model.coef_, target_idx)
        individ.intercept = model.intercept_

    @apply_decorator
    def apply(self, individ, *args, **kwargs) -> None:
        self._lr(individ)


class DEOptIndivid(GeneticOperatorIndivid):

    def __init__(self, params):
        super().__init__(params=params)

    def _preprocessing_data(self, individ):
        chromo = individ.get_structure()
        features = np.array(list(map(lambda token: token.value(self.params['grid']), chromo)))
        return features

    @staticmethod
    def _set_amplitudes_after_regression(individ, coefs):
        chromo = individ.get_structure()
        for idx, token in enumerate(chromo):
            token.set_param(token.param(name='Amplitude') * coefs[idx], name='Amplitude')

    @staticmethod
    def Q(w, *args):
        return np.linalg.norm(w @ args[0]) + args[1] * (1 / np.abs(w).sum() + np.abs(w).sum())

    @apply_decorator
    def apply(self, individ, *args, **kwargs) -> None:
        X = self._preprocessing_data(individ)
        bounds = [(-1, 1) for i in range(X.shape[0])]
        res = differential_evolution(self.Q, bounds, args=(X, 1), popsize=3)
        self._set_amplitudes_after_regression(individ, res.x)


class DelDuplicateTokensIndivid(GeneticOperatorIndivid):
    """
    Del all equivalent tokens in Individ's chromosome.
    """
    def __init__(self, params=None):
        super().__init__(params=params)

    @apply_decorator
    def apply(self, individ, *args, **kwargs):
        new_chromo = []
        flag = False
        for token in individ.structure:
            if token not in new_chromo:
                new_chromo.append(token)
            else:
                flag = True
        if flag:
            individ.structure = new_chromo

class CheckMandatoryTokensIndivid(GeneticOperatorIndivid):
    """
    Checking for the presence of required tokens in the chromosome and adding them.
    """
    def __init__(self, params=None):
        super().__init__(params=params)

    def _individ_mandatories(self, individ):
        mandatories = set()
        for token in individ.structure:
            if isinstance(token, ComplexToken):
                for token in token.structure:
                    mandatories.add(token.mandatory)
            else:
                mandatories.add(token.mandatory)
        return mandatories

    def _check_missing_mandatory_tokens_in(self, individ):
        individ_mandatories = self._individ_mandatories(individ)
        mandatories = set(map(lambda token: token.mandatory, self.params['tokens']))
        individ_mandatories.discard(0)
        mandatories.discard(0)
        diff = mandatories.difference(individ_mandatories)
        return list(filter(lambda token: token.mandatory in diff, self.params['tokens']))

    @apply_decorator
    def apply(self, individ, *args, **kwargs):
        missing_tokens = self._check_missing_mandatory_tokens_in(individ)
        if not missing_tokens:
            return
        terminal_tokens = list(filter(lambda token: not isinstance(token, ComplexToken), individ.structure))
        complex_tokens = list(filter(lambda token: isinstance(token, ComplexToken), individ.structure))
        for token in missing_tokens:
            # if terminal_tokens and complex_tokens:
            #     if np.random.uniform() < self.params['add_to_complex_prob']:
            #         complex_token = np.random.choice(complex_tokens)
            #         complex_token.add_subtoken(token.copy())
            #     else:
            #         individ.add_substructure(token.copy())
            # elif complex_tokens:
            #     complex_token = np.random.choice(complex_tokens)
            #     complex_token.add_subtoken(token.copy())
            # else:
            individ.add_substructure(token.copy())


class RestrictTokensIndivid(GeneticOperatorIndivid):
    def __init__(self, params=None):
        super().__init__(params=params)
        self._check_params()

    def _restrict_tokens(self, individ):
        # max_tokens = self.params['max_tokens']
        max_tokens = individ.max_tokens
        if max_tokens is None:
            return
        assert max_tokens >= 1, 'Tokens number must be equal or more than 1'
        if len(individ.structure) <= max_tokens:
            return
        individ.structure = copy(individ.structure[:max_tokens])
        # todo make probabilities

    @apply_decorator
    def apply(self, individ, *args, **kwargs):
        self._restrict_tokens(individ)


class LassoPopulation(GeneticOperatorPopulation):
    def __init__(self, params=None):
        super().__init__(params=params)

    def apply(self, population, *args, **kwargs):
        for individ in population.structure:
            # individ.apply_operator('LassoIndivid1Target')
            individ.apply_operator('TokenFitnessIndivid')
            individ.apply_operator('RestrictTokensIndivid')
            individ.apply_operator('LRIndivid1Target')
        return population


class RegularisationPopulation(GeneticOperatorPopulation):
    def __init__(self, params=None):
        super().__init__(params=params)

    def apply(self, population, *args, **kwargs):
        for individ in population.structure:
            if len(individ.structure) <= 1: # удаление индивида если помимо его константы нет других токенов
                population.del_substructure(individ)
                continue
            individ.apply_operator('CheckMandatoryTokensIndivid')
            individ.apply_operator('TokenFitnessIndivid')
            individ.apply_operator('DelDuplicateTokensIndivid')
            individ.apply_operator('RestrictTokensIndivid')
            # individ.apply_operator('LRIndivid1Target')

        return population

