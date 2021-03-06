#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:22:17 2021

@author: mike_ubuntu
"""

import numpy as np
import warnings
# from abc import ABC, abstractmethod, abstractproperty
from functools import reduce
import copy
import gc

from sklearn.linear_model import LinearRegression

import src.globals as global_var

from estar.src.factor import Factor
from src.supplementary import filter_powers, Population_Sort, form_label  # , memory_assesment
import src.moeadd.moeadd as moeadd


# import src.moeadd.moeadd_supplementary as moeadd_sup


def Check_Unqueness(obj, background):
    return not any([elem == obj for elem in background])


def normalize_ts(Input):
    #    print('normalize_ts Input:', Input)
    matrix = np.copy(Input)
    #    print(Matrix.shape)
    if np.ndim(matrix) == 0:
        raise ValueError('Incorrect input to the normalizaton: the data has 0 dimensions')
    elif np.ndim(matrix) == 0:
        return matrix
    else:
        for i in np.arange(matrix.shape[0]):
            std = np.std(matrix[i])
            if std != 0:
                matrix[i] = (matrix[i] - np.mean(matrix[i])) / std
            else:
                matrix[i] = 1
        return matrix


class Complex_Structure(object):
    def __init__(self, interelement_operator=np.add, *params):
        self.structure = None
        self.interelement_operator = interelement_operator

    def __eq__(self, other):
        assert type(other) == type(self)
        return (all(
            [any([other_elem == self_elem for other_elem in other.structure]) for self_elem in self.structure]) and
                all([any([other_elem == self_elem for self_elem in self.structure]) for other_elem in
                     other.structure]) and
                len(other.structure) == len(self.structure))

    def evaluate(self):
        assert len(self.structure) > 0, 'Attempt to evaluate an empty complex structure'
        if len(self.structure) == 1:
            return self.structure[0].evaluate()
        else:
            return reduce(lambda x, y: self.interelement_operator(x.evaluate(), y.evaluate()), self.structure)

    @property
    def name(self):
        pass


class Term(Complex_Structure):
    def __init__(self, tokens,
                 passed_term=None, max_factors_in_term=2, forbidden_tokens=None, interelement_operator=np.multiply):
        super().__init__(interelement_operator)
        self.tokens = tokens
        self.max_factors_in_term = max_factors_in_term

        if type(passed_term) == type(None):
            self.Randomize(forbidden_tokens)
        else:
            self.Defined(passed_term)

        if type(global_var.tensor_cache) != type(None):
            self.use_cache()
        #        if type(global_var.grid_cache) != type(None):
        #            self.use_grid_cache()
        self.clear_value()  # key - state of normalization, value - if the variable is saved in cache

    @property
    def cache_label(self):
        if len(self.structure) > 1:
            structure_sorted = sorted(self.structure, key=lambda x: x.cache_label)
            cache_label = reduce(form_label, structure_sorted, '')
        else:
            cache_label = self.structure[0].cache_label
        return cache_label

    def use_cache(self):
        self.cache_linked = True
        for idx, _ in enumerate(self.structure):
            if not self.structure[idx].cache_linked:
                self.structure[idx].use_cache()

    #
    #    def use_grids_cache(self): #!
    #        self.grid_cache_linked = True
    #        for idx, _ in enumerate(self.structure):
    #            if not self.structure[idx].cache_linked:
    #                self.structure[idx].use_cache()

    def Defined(self, passed_term):
        self.structure = []

        if type(passed_term) == list or type(passed_term) == tuple:
            for i, factor in enumerate(passed_term):
                if type(factor) == str:
                    token_family = [token_family for token_family in self.tokens if factor in token_family.tokens][0]
                    self.structure.append(Factor(factor, token_family, randomize=True))
                elif type(factor) == Factor:
                    self.structure.append(factor)
                else:
                    raise ValueError(
                        'The structure of a term should be declared with str or src.factor.Factor obj, instead got',
                        type(factor))
        else:  # ????????????, ???????? ???????????????? ???????? 1 ??????????
            if type(passed_term) == str:
                token_family = [token_family for token_family in self.tokens if passed_term in token_family.tokens][0]
                self.structure.append(Factor(passed_term, token_family, randomize=True))
            elif type(passed_term) == Factor:
                self.structure.append(passed_term)
            else:
                raise ValueError(
                    'The structure of a term should be declared with str or src.factor.Factor obj, instead got',
                    type(passed_term))

    def Randomize(self, forbidden_factors=None):
        meaningful_tokens = [token_family for token_family in self.tokens if token_family.status['meaningful']]
        total_meaningful = np.sum([len(token.tokens) for token in meaningful_tokens])

        if len(meaningful_tokens) == 0:
            raise ValueError('No token families are declared as meaningful for the process of the system search')
        # ?? ???????????? ?????????????????? ?????????????? 1 ??????????, ??????????. ???????????????????????? ????????????????
        #        if len(mandatory_tokens) > self.max_factors_in_term:
        #            raise ValueError('Number of mandatory tokens for term exceeds allowed number of factors')
        factors_num = np.random.randint(1, self.max_factors_in_term + 1)
        #        print('creating factors', factors_num)

        while True:
            self.occupied_tokens_labels = []
            self.structure = []
            # ?? ???????? ?????????????????????? ???????????????????????????? ???????????? 1 ???????????????????????? ?????????? ?? ??????????????????, ???????????? ???????????????????????????? ?????????? ???????? ?????????????????? ?????? ????????????????????
            selected_meaningful_family = np.random.choice(meaningful_tokens,
                                                          p=[len(token.tokens) / float(total_meaningful) for token in
                                                             meaningful_tokens])
            selected_label = np.random.choice(selected_meaningful_family.tokens)
            if selected_meaningful_family.status[
                'unique_specific_token']:  # 'single' - ?????????? ???? ?????????? ?????????????????????? (???????? ?? ?????????????? ??????-????)
                self.occupied_tokens_labels.append(selected_label)
            elif selected_meaningful_family.status[
                'unique_token_type']:  # 'unique_token' - ?? ?????????????????? ???? ?????????? ???????? 2 ????-???????? ???????????????????? ????????
                self.occupied_tokens_labels.extend([token_label for token_label in selected_meaningful_family.tokens])

            self.structure.append(Factor(selected_label, selected_meaningful_family,
                                         randomize=True))  # ?????????????? ????????. ???????????????????????????? ???????????????????? ?????????????????? ?? ???????? ?? ?????????????? ????????????????

            for i in np.arange(1, factors_num):
                total_tokens_num = np.sum([len(token_family.tokens) for token_family in self.available_tokens],
                                          dtype=np.float32)
                selected_token = np.random.choice(self.available_tokens,
                                                  p=[len(token_family.tokens) / total_tokens_num for token_family in
                                                     self.available_tokens])
                selected_label = np.random.choice(selected_token.tokens)

                if selected_token.status[
                    'unique_specific_token']:  # 'unique_token_type' - ?????????? ???? ?????????? ?????????????????????? (???????? ?? ?????????????? ??????-????)
                    self.occupied_tokens_labels.append(selected_label)
                elif selected_token.status[
                    'unique_token_type']:  # 'unique_specific_token' - ?? ?????????????????? ???? ?????????? ???????? 2 ????-???????? ???????????????????? ????????
                    self.occupied_tokens_labels.extend([token_label for token_label in selected_token.tokens])

                self.structure.append(Factor(selected_label, selected_token, randomize=True))
            #                self.structure[i].Set_parameters(random = True)
            self.structure = filter_powers(self.structure)
            if type(forbidden_factors) == type(None):
                break
            elif all([(Check_Unqueness(factor, forbidden_factors) or not factor.token.status['unique_for_right_part'])
                      for factor in self.structure]):
                break

    def Remove_Dublicated_Factors(self, target, background_structure):
        raise DeprecationWarning('To be deleted')
        cleared_num = 0
        cleared_token_families = []
        structure_cleared = []
        for factor_idx in range(len(self.structure)):
            if all([(elem != self.structure[factor_idx] or not elem.token.status['unique_for_right_part']) for elem in
                    target.structure]):
                structure_cleared.append(self.structure[factor_idx])
            else:
                cleared_token_families.append(self.structure[factor_idx].token.type)
                cleared_num += 1

        token_selection = self.available_tokens
        filling_try = 0
        while True:
            filling_try += 1

            cleared_token_families_copy = copy.copy(cleared_token_families)
            if filling_try == 10:
                cleared_num += 1
                cleared_token_families_copy.append(np.random.choice(cleared_token_families))

            if filling_try == 100:
                print('background:', [token.text_form for token in background_structure])
                print('self:', self.text_form, cleared_num)
                warnings.warn(
                    'Algorithm is having issues with creation of unique structure: reduce length of equation or increase token pool')

            structure_filled = copy.deepcopy(structure_cleared)
            iter_selection_pool = copy.deepcopy(token_selection)
            for i in np.arange(cleared_num):
                while True:
                    iter_selection_pool_copy = copy.copy(iter_selection_pool)
                    cleared_token_families_copy_2 = copy.copy(cleared_token_families_copy)

                    total_tokens_num = np.sum([len(token_family.tokens) for token_family in
                                               [token_family_loc for token_family_loc in iter_selection_pool_copy if
                                                token_family_loc.type in cleared_token_families_copy]],
                                              dtype=np.float32)
                    try:
                        selected_token = np.random.choice(
                            [token_family_loc for token_family_loc in iter_selection_pool_copy if
                             token_family_loc.type in cleared_token_families_copy_2],
                            p=[len(token_family.tokens) / total_tokens_num for token_family in
                               [token_family_loc for token_family_loc in iter_selection_pool_copy if
                                token_family_loc.type in cleared_token_families_copy_2]])
                    except ValueError:
                        print('cleared tokens:', cleared_token_families_copy_2, cleared_token_families)
                        print()
                        print('current pool:', [token_family_loc for token_family_loc in iter_selection_pool_copy if
                                                token_family_loc.type in cleared_token_families_copy_2])
                        print('occupied tokens:', self.occupied_tokens_labels)
                        print('available:', [token_family.tokens for token_family in token_selection])
                        raise ValueError(
                            'The framework had issues with creating filtering the tokens: the selection pool is emptied.') from None

                    try:
                        selected_label = np.random.choice(selected_token.tokens)
                    except ValueError:
                        print('selected label:', selected_token.type, selected_token.tokens)
                        print('cleared tokens:', cleared_token_families_copy_2, cleared_token_families)
                        print('current pool:',
                              [token_family_loc.tokens for token_family_loc in iter_selection_pool_copy if
                               token_family_loc.type in cleared_token_families_copy_2])
                        print('occupied tokens:', self.occupied_tokens_labels)
                        print('available:', [token_family.tokens for token_family in token_selection])
                        print('self:', self.name)
                        for term in background_structure:
                            print(term.name)
                        raise ValueError(
                            'The framework had issues with selecting the token. Probable cause: lack of initialized tokens') from None

                    cleared_token_families_copy_2.remove(selected_token.type)
                    iter_selection_pool_copy[iter_selection_pool_copy.index(selected_token)].tokens.remove(
                        selected_label)

                    temp_token = Factor(selected_label, selected_token, randomize=True)
                    #                    parameter_selection = copy.deepcopy(selected_token.token_params)
                    #                    for param, interval in selected_token.token_params.items():
                    #                        if interval[0] == interval[1]:
                    #                            parameter_selection[param] = interval[1]
                    #                            continue
                    #                        if param == 'power':
                    #                            parameter_selection[param] = 1
                    #                            continue
                    #                        parameter_selection[param] = (np.random.randint(interval[0], interval[1])
                    #                                                      if isinstance(interval[0], int) else np.random.uniform(interval[0], interval[1]))
                    #                    temp_token.Set_parameters(**parameter_selection)
                    #                    if all([(Check_Unqueness(factor, target.structure) or factor.token.status['unique_for_right_part']) for factor in new_term.structure]):
                    if (Check_Unqueness(temp_token, target.structure) or not temp_token.token.status[
                        'unique_for_right_part']):
                        structure_filled.append(temp_token)
                        break

            new_term = copy.deepcopy(self)
            new_term.structure = structure_filled
            if Check_Unqueness(new_term,
                               background_structure):  # and all([(Check_Unqueness(factor, target.structure) or factor.token.status['unique_for_right_part']) for factor in new_term.structure]):
                del new_term
                break
        self.structure = structure_filled
        self.clear_value()
        for idx, _ in enumerate(self.structure):
            #            if not self.structure[idx].cache_linked:
            self.structure[idx].use_cache()

    def evaluate(self, normalize=True):
        assert type(global_var.tensor_cache) != type(None), 'Currently working only with connected cache'
        #        print('Normalized state of current equaton: ', normalize)

        if self.saved[normalize]:
            value = global_var.tensor_cache.get(self.cache_label, normalize, self.saved_as[normalize])
            value = value.reshape(value.size)
            return value
        else:
            self.prev_normalized = normalize
            value = super().evaluate()
            #            print(self.name, ' - ')
            #            print(value.shape)
            if normalize and np.ndim(value) != 1:
                value = normalize_ts(value)
            elif normalize and np.ndim(value) == 1 and np.std(value) != 0:
                value = (value - np.mean(value)) / np.std(value)
            elif normalize and np.ndim(value) == 1 and np.std(value) == 0:
                value = (value - np.mean(value))
            if np.all([len(factor.params) == 1 for factor in self.structure]):
                #                print(self.cache_label)
                self.saved[normalize] = global_var.tensor_cache.add(self.cache_label, value,
                                                                    normalize)  # ?????????? ?????????????????? ??????????????: ????????????????????/???????????????? ?????????????????????????????? ????????????
                if self.saved[normalize]: self.saved_as[normalize] = self.cache_label
            value = value.reshape(value.size)
            return value

    def clear_value(self):
        self.saved = {True: False, False: False}
        self.saved_as = {True: None, False: None}

    def Reset_occupied_tokens(self):
        occupied_tokens_new = []
        for factor in self.structure:
            for token_family in self.tokens:
                if factor.label in token_family.tokens and token_family.status['unique_token_type']:
                    occupied_tokens_new.extend([token for token in token_family.tokens])
                elif factor.label in token_family.tokens and token_family.status['unique_specific_token']:
                    occupied_tokens_new.append(factor.label)
        self.occupied_tokens_labels = occupied_tokens_new

    @property
    def available_tokens(self):
        available_tokens = []
        for token in self.tokens:
            if not all([label in self.occupied_tokens_labels for label in token.tokens]):
                token_new = copy.deepcopy(token)
                token_new.tokens = [label for label in token.tokens if label not in self.occupied_tokens_labels]
                available_tokens.append(token_new)
        return available_tokens

    @property
    def total_params(self):
        return max(sum([len(element.params) - 1 for element in self.structure]), 1)

    @property
    def name(self):
        form = ''
        for token_idx in range(len(self.structure)):
            form += self.structure[token_idx].name
            if token_idx < len(self.structure) - 1:
                form += ' * '
        return form

    @property
    def latex_form(self):
        form = r""

    # TODO Duplicated code (line 52-55)
    def __eq__(self, other):
        assert type(other) == type(self)
        return (all(
            [any([other_elem == self_elem for other_elem in other.structure]) for self_elem in self.structure]) and
                all([any([other_elem == self_elem for self_elem in self.structure]) for other_elem in
                     other.structure]) and
                len(other.structure) == len(self.structure))


class Equation(Complex_Structure):
    def __init__(self, tokens, basic_structure, terms_number=6, max_factors_in_term=2,
                 interelement_operator=np.add):  # eq_weights_eval

        """

        Class for the single equation for the dynamic system.
            
        attributes:
            structure : list of Term objects \r\n
            List, containing all terms of the equation; first 2 terms are reserved for constant value and the input function;
        
            target_idx : int \r\n
            Index of the target term, selected in the Split phase;
        
            target : 1-d array of float \r\n
            values of the Term object, reshaped into 1-d array, designated as target for application in sparse regression;
            
            features : matrix of float \r\n
            matrix, composed of terms, not included in target, value columns, designated as features for application in sparse regression;
        
            fitness_value : float \r\n
            Inverse value of squared error for the selected target 2function and features and discovered weights; 
        
            estimator : sklearn estimator of selected type \r\n
        
        parameters:

            Matrix of derivatives: first axis through various orders/coordinates in order: ['1', 'f', all derivatives by one coordinate axis
            in increasing order, ...]; second axis: time, further - spatial coordinates;

            tokens : list of strings \r\n
            Symbolic forms of functions, including derivatives;

            terms_number : int, base value of 6 \r\n
            Maximum number of terms in the discovered equation; 

            max_factors_in_term : int, base value of 2\r\n
            Maximum number of factors, that can form a term (e.g. with 2: df/dx_1 * df/dx_2)

        """
        super().__init__(interelement_operator)

        self.n_immutable = len(basic_structure)
        self.tokens = tokens
        self.structure = []
        self.terms_number = terms_number;
        self.max_factors_in_term = max_factors_in_term
        self.operator = None
        if (terms_number < self.n_immutable):
            raise Exception(
                'Number of terms ({}) is too low to even contain all of the pre-determined ones'.format(terms_number))

        self.structure.extend(
            [Term(self.tokens, passed_term=label, max_factors_in_term=self.max_factors_in_term) for label in
             basic_structure])

        for i in range(len(basic_structure), terms_number):
            check_test = 0
            while True:
                check_test += 1
                new_term = Term(self.tokens, max_factors_in_term=self.max_factors_in_term, passed_term=None)
                if Check_Unqueness(new_term, self.structure):
                    break
            self.structure.append(new_term)

        for idx, _ in enumerate(self.structure):
            #            if type(tokens[0].cache) != type(None):
            self.structure[idx].use_cache()
        #            self.structure[idx].use_grid_cache()
        #                self.cache = tokens[0].cache
        self.reset_eval_state()

    def select_target_idx(self, separate_vars=[], operator=None, target_idx_fixed=None):

        '''
        
        Separation of target term from features & removal of factors, that are in target, from features
        
        '''
        assert (type(self.operator) != type(None) or type(operator) != type(None)) or type(target_idx_fixed) != type(
            None)
        if type(self.operator) == type(None) and type(operator) != type(None): self.operator = operator

        if type(target_idx_fixed) == type(None):
            max_fitness = 0
            max_idx = 0
            for target_idx, target_term in enumerate(self.structure):
                self.target_idx = target_idx
                self.operator.get_fitness(self)
                if self.described_variables in separate_vars:
                    self.penalize_fitness(coeff=0.)
                if self.fitness_value > max_fitness:
                    max_fitness = self.fitness_value
                    max_idx = target_idx
            self.target_idx = max_idx
            self.operator.get_fitness(self)
            assert self.fitness_value == max_fitness
        else:
            self.target_idx = target_idx_fixed

        # self.target_idx = target_idx if type(target_idx) != type(None) else np.random.randint(low = 0, high = len(
        # self.structure)-1)
        self.target_distingusied = False

    #        for feat_idx in range(len(self.structure)):
    #            if self.target_idx != feat_idx:
    #                self.structure[feat_idx].Remove_Dublicated_Factors(self.structure[self.target_idx], self.structure[:feat_idx]+self.structure[feat_idx+1:])
    #            else:
    #                continue

    def check_split_correctness(self):  # Refactor for needs of system discovery
        if True:
            pass
        else:
            target = self.structure[self.target_idx]
            for term_idx in range(len(self.structure)):
                if term_idx == self.target_idx: continue
                for factor in self.structure[term_idx].structure:
                    if factor.token.status['unique_for_right_part'] and any(
                            [factor == rp_factor for rp_factor in target.structure]):
                        print('In target:', target.name)
                        print('In other structure discovered:',
                              [factor.label for factor in self.structure[term_idx].structure])
                        print([term.name for term in self.structure])
                        raise ValueError('Forbidden factor ')

    @property
    def forbidden_token_labels(self):
        target_symbolic = [factor.label for factor in self.structure[self.target_idx].structure]
        forbidden_tokens = set()

        for token_family in self.tokens:
            for token in token_family.tokens:
                if token in target_symbolic and token_family.status['unique_for_right_part']:
                    forbidden_tokens.add(token)
        return forbidden_tokens

    def evaluate(self, normalize=True, return_val=False):
        self._target = self.structure[self.target_idx].evaluate(normalize)
        feature_indexes = list(range(len(self.structure)))
        feature_indexes.remove(self.target_idx)
        for feat_idx in range(len(feature_indexes)):
            if feat_idx == 0:
                self._features = self.structure[feature_indexes[feat_idx]].evaluate(normalize)
            elif feat_idx != 0:  # and self.target_idx != feat_idx:
                temp = self.structure[feature_indexes[feat_idx]].evaluate(normalize)
                self._features = np.vstack([self._features, temp])
            else:
                continue
        temp_feats = np.vstack([self._features, np.ones(self._features.shape[1])])
        self._features = np.transpose(self._features);
        temp_feats = np.transpose(temp_feats)
        if return_val:
            self.prev_normalized = normalize
            if normalize:
                #                value = self._target - np.add(*[np.multiply(weight, temp_feats[:,feature_idx])
                #                                                    for feature_idx, weight in np.ndenumerate(self.weights_internal)])
                value = self._target - reduce(lambda x, y: np.add(x, y),
                                              [np.multiply(weight, temp_feats[:, feature_idx])
                                               for feature_idx, weight in np.ndenumerate(self.weights_internal)])

            else:
                #                print(self._features.shape, temp_feats.shape)
                #                print('mult test', [np.multiply(weight, temp_feats[:, feature_idx])
                #                                                    for feature_idx, weight in np.ndenumerate(self.weights_final)])
                #                test_var_1 = np.multiply(self.weights_final[0], temp_feats[:,0])
                #                test_var_2 = np.multiply(self.weights_final[1], temp_feats[:,1])
                #                test_var_3 = np.multiply(self.weights_final[2], temp_feats[:,2])
                #
                #                test_res = reduce(lambda x,y: np.add(x, y), [np.multiply(weight, temp_feats[:,feature_idx])
                #                                                    for feature_idx, weight in np.ndenumerate(self.weights_final)])
                elem1 = np.expand_dims(self._target, axis=1)

                value = np.add(elem1,
                               - reduce(lambda x, y: np.add(x, y), [np.multiply(weight, temp_feats[:, feature_idx])
                                                                    for feature_idx, weight in
                                                                    np.ndenumerate(self.weights_final)]))
            #                print('value:', value, value.shape)

            return value, self._target, self._features
        else:
            return None, self._target, self._features

    def reset_eval_state(self):
        #        self.evaluated = False
        self.weights_internal_evald = False
        self.weights_final_evald = False

    #        self.prev_normalized = False

    @property
    def fitness_value(self):
        return self._fitness_value

    @fitness_value.setter
    def fitness_value(self, val):
        self._fitness_value = val

    def penalize_fitness(self, coeff=1.):
        self._fitness_value = self._fitness_value * coeff

    @property
    def L0_norm(self):
        return np.count_nonzero(self.weights_internal)

    @property
    def weights_internal(self):
        if self.weights_internal_evald:
            return self._weights_internal
        else:
            raise AttributeError('Internal weights called before initialization')

    @weights_internal.setter
    def weights_internal(self, weights):
        self._weights_internal = weights
        self.weights_internal_evald = True

    @property
    def weights_final(self):
        if self.weights_final_evald:  # ?????????????????? ???????????????????? ?????????? ?????????? ?????????????????? ?????????????????? ??????????????????
            return self._weights_final
        else:
            self._weights_final = Get_true_coeffs(self.tokens, self)
            self.weights_final_evald = True
            del self._features, self._target
            return self._weights_final

    @property
    def latex_form(self):
        form = r""
        for term_idx in range(len(self.structure)):
            if term_idx != self.target_idx:
                form += str(self.weights_final[term_idx]) if term_idx < self.target_idx else str(
                    self.weights_final[term_idx - 1])
                form += ' * ' + self.structure[term_idx].latex_form + ' + '
        form += str(self.weights_final[-1]) + ' = ' + self.structure[self.target_idx].text_form
        return form

    @property
    def text_form(self):
        form = ''
        for term_idx in range(len(self.structure)):
            if term_idx != self.target_idx:
                form += str(self.weights_final[term_idx]) if term_idx < self.target_idx else str(
                    self.weights_final[term_idx - 1])
                form += ' * ' + self.structure[term_idx].name + ' + '
        form += str(self.weights_final[-1]) + ' = ' + self.structure[self.target_idx].text_form
        return form

    @property
    def described_variables(self):
        eps = 1e-7
        described = set()
        for term_idx, term in enumerate(self.structure):
            if term_idx == self.target_idx:
                described.update({factor.token.type for factor in term.structure})
            else:
                weight_idx = term_idx if term_idx < term_idx else term_idx - 1
                if np.abs(self.weights_final[weight_idx]) > eps:
                    #                    print('to described', self.weights_final[weight_idx], {factor.token.type for factor in term.structure})
                    described.update({factor.token.type for factor in term.structure})
        described = frozenset(described)
        return described


def Get_true_coeffs(tokens, equation):  # ???? ???????????? ?????? ????, ?????? ?????????????????? ???????? - ?????? ??????????????????
    assert equation.weights_internal_evald, 'Trying to calculate final weights before evaluating intermeidate ones (no sparcity).'
    target = equation.structure[equation.target_idx]

    equation.check_split_correctness()

    #    print(type(target.value))
    target_vals = target.evaluate(False)
    features_vals = []
    nonzero_features_indexes = []
    for i in range(len(equation.structure)):
        if i == equation.target_idx:
            continue
        idx = i if i < equation.target_idx else i - 1
        if equation.weights_internal[idx] != 0:
            features_vals.append(equation.structure[i].evaluate(False))
            nonzero_features_indexes.append(idx)

    #    print('Indexes of nonzero elements:', nonzero_features_indexes)
    if len(features_vals) == 0:
        return np.zeros(len(
            equation.structure))  # Bind_Params([(token.label, token.params) for token in target.structure]), [('0', 1)]

    features = features_vals[0]
    if len(features_vals) > 1:
        for i in range(1, len(features_vals)):
            features = np.vstack([features, features_vals[i]])
    features = np.vstack([features, np.ones(features_vals[0].shape)])  # ?????????????????? ?????????????????????? ????????
    features = np.transpose(features)

    estimator = LinearRegression(fit_intercept=False)
    try:
        estimator.fit(features, target_vals)
    except ValueError:
        features = features.reshape(-1, 1)
        estimator.fit(features, target_vals)

    valueable_weights = estimator.coef_
    weights = np.zeros(len(equation.structure))
    for weight_idx in range(len(weights) - 1):
        if weight_idx in nonzero_features_indexes:
            weights[weight_idx] = valueable_weights[nonzero_features_indexes.index(weight_idx)]
    weights[-1] = valueable_weights[-1]
    #    print('weights check:', weights, equation.weights_internal)
    return weights


class SoEq(Complex_Structure, moeadd.moeadd_solution):
    def __init__(self, token_families, terms_number, max_factors_in_term, sparcity=None, eq_search_iters=100):
        #        assert type(sparcity) != type(None)
        self.tokens_indep = [family for family in token_families if family.status['meaningful']]
        self.tokens_dep = [family for family in token_families if not family.status['meaningful']]
        self.equation_number = len(self.tokens_indep)

        if type(sparcity) != None: self.vals = sparcity
        self.max_terms_number = terms_number;
        self.max_factors_in_term = max_factors_in_term
        self.moeadd_set = False;
        self.eq_search_operator_set = False  # ; self.evaluated = False
        self.def_eq_search_iters = eq_search_iters  # None

    def set_eq_search_evolutionary(self, evolutionary):
        assert type(evolutionary.coeff_calculator) != type(
            None), 'Defined evolutionary operator lacks coefficient calculator'
        self.eq_search_evolutionary_operator = evolutionary
        self.eq_search_operator_set = True

    def create_equations(self, population_size=16, sparcity=None, eq_search_iters=None):
        #        if type(eq_search_iters) == type(None) and type(self.def_eq_search_iters) == type(None):
        #            raise ValueError('Number of iterations is not defied both in method parameter or in object attribute')
        assert self.eq_search_operator_set

        if type(eq_search_iters) == type(None): eq_search_iters = self.def_eq_search_iters
        if type(sparcity) == type(None):
            sparcity = self.vals
        else:
            self.vals = sparcity
        self.population_size = population_size
        self.structure = [];
        self.eq_search_iters = eq_search_iters
        token_selection = copy.deepcopy(self.tokens_indep)

        self.vars_to_describe = {token_family.type for token_family in self.tokens_dep}
        self.vars_to_describe = self.vars_to_describe.union({token_family.type for token_family in self.tokens_indep})
        self.separated_vars = set()

        for eq_idx in range(self.equation_number):
            current_tokens = token_selection + self.tokens_dep
            self.eq_search_evolutionary_operator.set_sparcity(sparcity_value=self.vals[eq_idx])
            cur_equation, cur_eq_operator_error = self.optimize_equation(current_tokens,
                                                                         self.eq_search_evolutionary_operator,
                                                                         self.population_size, eq_search_iters,
                                                                         separate_vars=self.separated_vars)
            self.vars_to_describe.difference_update(cur_equation.described_variables)
            self.separated_vars.add(frozenset(cur_equation.described_variables))

            self.structure.append(cur_equation)
            #            self.single_vars_in_equation.update()
            #            cache.clear(full = False)
            global_var.tensor_cache.change_variables(cur_eq_operator_error)
        #            for idx, _ in enumerate(token_selection):
        #                 token_selection[idx].change_variables(cur_eq_operator_error)

        obj_funs = [self.evaluate, ] + [eq.L0_norm for eq in self.structure]
        moeadd.moeadd_solution.__init__(self, self.vals, obj_funs)  # , self) super(
        self.moeadd_set = True

    def optimize_equation(self, tokens, operator, population_size, eq_search_iters, basic_terms: list = [],
                          separate_vars: set = None):
        population = [Equation(tokens, basic_terms, self.max_terms_number, self.max_factors_in_term)
                      for i in range(population_size)]
        for idx, equation in enumerate(population):
            print('initial population, idx ', idx, len(equation.structure))
            equation.select_target_idx(separate_vars, operator)
            equation.check_split_correctness()
            #            [print(term.text_form) for term in equation.structure]
            #            print('equation target:', equation.target_idx, equation.structure[equation.target_idx].text_form)
            operator.get_fitness(equation)
            if equation.described_variables in separate_vars:
                equation.penalize_fitness(coeff=0.)

        for idx in range(eq_search_iters):
            print('Equation optimiation iteration', idx)
            strict_restrictions = False if idx < eq_search_iters - 1 else True
            #            print(population[0].text_form, population[0].structure[population[0].target_idx].text_form)
            population = self.equation_opt_iteration(population, self.eq_search_evolutionary_operator,
                                                     population_size, idx, separate_vars, strict_restrictions)
            for equation in population:
                if equation.described_variables in separate_vars:
                    equation.penalize_fitness(coeff=0.)
        population = Population_Sort(population)
        del population[1:]

        return population[0], population[0].evaluate(normalize=False, return_val=True)[0]

    @staticmethod
    def equation_opt_iteration(population, evol_operator, population_size, iter_index, separate_vars,
                               strict_restrictions=True):
        for equation in population:
            if equation.described_variables in separate_vars:
                equation.penalize_fitness(coeff=0.)
        population = Population_Sort(population)
        population = population[:population_size]
        print(iter_index, population[0].fitness_value, population[0].described_variables, separate_vars,
              population[0].described_variables in separate_vars)
        print('Cache size', np.round(global_var.tensor_cache.consumed_memory / 1024 ** 2, decimals=3), 'MB')
        gc.collect()
        #        if iter_index == 2:
        #            memory_assesment()
        #        time.sleep(10)
        #        prev_population = copy.deepcopy(population) # copy.deepcopy?
        population = evol_operator.apply(population, separate_vars)
        return population

    def __eq__(self, other):
        assert self.moeadd_set, 'The structure of the equation is not defined, therefore no moeadd operations can be called'
        epsilon = 1e-6
        if isinstance(other, type(self)):
            return np.all(np.abs(self.vals - other.vals) < epsilon)
        else:
            return NotImplemented

    def evaluate(self, normalize=True):
        #            self.evaluated = True
        value = np.sum([equation.value(normalize) for equation in self.population])
        return value

    @property
    def obj_fun(self):
        return self.obj_funs

    def __call__(self):
        assert self.moeadd_set, 'The structure of the equation is not defined, therefore no moeadd operations can be called'
        return self.obj_fun

    @property
    def text_form(self):
        form = ''
        if len(self.structure) > 1:
            for eq_idx, equation in enumerate(self.structure):
                if eq_idx == 0:
                    form += '/ ' + equation.text_form + '\n'
                elif eq_idx == len(self.structure) - 1:
                    form += '\ ' + equation.text_form + '\n'
                else:
                    form += '| ' + equation.text_form + '\n'
        else:
            form += equation.text_form + '\n'
        return form

    @property
    def latex_form(self):
        form = r"\begin{eqnarray*}"
        for equation in self.structure:
            form += equation.latex_form + r", \\ "
        form += r"\end{eqnarray*}"

    def __hash__(self):
        return hash(tuple(self.vals))
