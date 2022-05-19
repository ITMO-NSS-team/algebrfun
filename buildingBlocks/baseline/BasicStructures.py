"""
Содержит базовые абстрактные структуры, от которых наследуются основные сущности алгоритма.
"""
from copy import deepcopy
from typing import Optional, Union

import numpy as np


class Token:
    """
    A token is an entity that has some meaning in the context
    of a given task, and encapsulates information that is sufficient to work with it.
    """

    def __init__(self, number_params: int = 0, params_description: dict = None,
                 params: np.ndarray = None, name_: str = None):
        self._number_params = number_params
        if params_description is None:
            params_description = {}
        self.params_description = params_description
        if params is None:
            self.params = np.array([np.zeros(self._number_params)])
        else:
            self.params = np.array(params, dtype=float)
        if name_ is None:
           name_ = type(self).__name__
        self.name_ = name_

    def value(self, grid: np.ndarray) -> np.ndarray:
        """
        Return value of the token in the context of the task.

        Parameters
        ----------
        grid:
            The grid on which the value is calculated.
        """
        raise NotImplementedError("Token must have method value()")

    def name(self, with_params=False):
        return self.name_

    def copy(self):
        return deepcopy(self)

    # Methods for work with params and its descriptions
    @property
    def params_description(self):
        return self._params_description

    @params_description.setter
    def params_description(self, params_description: dict):
        """
        Params_description is dictionary of dictionaries for describing numeric parameters of the token.
            Must have the form like:
            {
                parameter_index=0: dict(name='name', bounds=(min_value, max_value)[, ...]),
                ...
            }
        Params_description must contain all fields for work in current tokens that will be checked by
        method 'self.check_params_description()'.

        Parameters
        ----------
        params_description: dict
            Dictionary with description for each parameter
        """
        self._params_description = params_description
        self.check_params_description()

    def check_params_description(self):
        """
        Check params_description for requirements for current token.
        """
        recomendations = "\nUse methods 'params_description.setter' or 'set_descriptor' to change params_descriptions"
        assert type(self._params_description) == dict, "Invalid params_description structure," \
                                                       " must be a dictionary of dictionaries" + recomendations
        assert len(self._params_description) == self._number_params, "The number of parameters does not" \
                                                                     " match the number of descriptors" + recomendations
        for key, value in self._params_description.items():
            assert type(value) == dict, "Invalid params_description structure, must be a dictionary of dictionaries"
            assert 'name' in value.keys(), "Key 'name' must be in the nested" \
                                           " dictionary for each parameter" + recomendations

    def set_descriptor(self, key: int, descriptor_name: str, descriptor_value):
        try:
            self._params_description[key][descriptor_name] = descriptor_value
        except KeyError:
            print('There is no parameter with such index/descriptor')
        self.check_params_description()

    def get_key_use_params_description(self, descriptor_name: str, descriptor_value):
        for key, value in self._params_description.items():
            if value[descriptor_name] == descriptor_value:
                return key
        raise KeyError('There is no descriptor with name {}'.format(descriptor_name))

    def get_descriptor_foreach_param(self, descriptor_name: str) -> list:
        ret = [None for _ in range(self._number_params)]
        for key, value in self._params_description.items():
            try:
                ret[key] = value[descriptor_name]
            except KeyError:
                KeyError('There is no descriptor with name {}'.format(descriptor_name))
        return ret

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params: np.ndarray):
        # self._params = np.array(params, dtype=float)
        self._params = params
        self.check_params()

    def check_params(self):
        isinstance(self._params, np.ndarray)
        print("assert shape of params", self._params.shape[-1], self._number_params)
        assert self._params.shape[-1] == self._number_params, "The number of parameters does not match the length of params array"\
                                                  + "\nUse methods 'params.setter' or 'set_param' to change params"
        # TODO проверка на количество переменных в уравнениии self._params.shape[0]

    def param(self, name=None, idx=None):
        try:
            idx = idx if name is None else self.get_key_use_params_description('name', name)
        except KeyError:
            raise KeyError('There is no parameter with this name')
        try:
            return self.params[:, idx] # !!!!!
        except IndexError:
            raise IndexError('There is no parameter with this index')

    def set_param(self, param, name=None, idx=None):
        try:
            idx = idx if name is None else self.get_key_use_params_description('name', name)
        except KeyError:
            raise KeyError('"{}" have no parameter with name "{}"'.format(self, name))
        try:
            print("Gg", self, idx, self._params[0][idx], param)
            try:
                for i in range(len(param)):
                    self._params[i][idx] = param[i] # разобраться, так как не все параметры расширяемы!!!!
            except:
                self._params[0][idx] = param
        except IndexError:
            raise IndexError('"{}" have no parameter with index "{}"'.format(self, idx))
        self.check_params()


class ComplexStructure:
    """Базовый абстрактный класс, предоставляющий методы работы со списком подструктур данной структуры"""

    def __init__(self, structure: list = None):
        if structure is None:
            structure = []
        self.structure = structure

    def _init_structure(self, structure: list = None):
        if structure is None:
            structure = []
        self.structure = structure

    @property
    def structure(self) -> list:
        return self._structure

    @structure.setter
    def structure(self, structure: list) -> None:
        assert type(structure) == list, "structure must be a list"
        self._structure = structure

    def get_substructure(self, idx: int):
        return self.structure[idx]

    def set_substructure(self, substructure, idx: int) -> None:
        self.structure[idx] = substructure

    def add_substructure(self, substructure, idx: int = -1) -> None:
        if idx is None or idx == -1 or idx == len(self.structure):
            try:
                self.structure.extend(substructure)
            except TypeError:
                self.structure.append(substructure)
        else:
            tmp_structure = self.structure[:idx]
            try:
                tmp_structure.extend(substructure)
            except TypeError:
                tmp_structure.append(substructure)
            tmp_structure.extend(self.structure[idx:])
            self.structure = tmp_structure

    def del_substructure(self, substructure):
        #TODO обработать исключения и работу с массивами
        self.structure.remove(substructure)


class EvolutionaryStructure(ComplexStructure):
    """Предостовляет дополнительный функционал для работы с генетическими операторами"""

    def __init__(self, structure: list = None, genetic_operators: dict = None):
        super().__init__(structure)
        if genetic_operators is None:
            genetic_operators = {}
        self.genetic_operators = genetic_operators

    @property
    def genetic_operators(self):
        return self._genetic_operators

    @genetic_operators.setter
    def genetic_operators(self, genetic_operators):
        self._genetic_operators = genetic_operators

    def apply_operator(self, name: str, *args, **kwargs):
        raise NotImplementedError('Not Implemented method "apply_operator()" for Evolutionary Structure')


class GeneticOperator:
    """
    Abstract class (need to implement method 'apply').
    Operator is applied to some object and change its properties.
    Work inplace (object is being changed by it applying).
    """
    def __init__(self, params: dict = None):
        """

        Parameters
        ----------
        params: dict
            Parameters for operator work.
        """
        if params is None:
            params = {}
        self.params = params

    def _check_params(self, *keys) -> None:
        params_keys = self.params.keys()
        for key in keys:
            assert key in params_keys, "Key {} must be in {}.params".format(key, type(self).__name__)

    def apply(self, target, *args, **kwargs) -> None:
        raise NotImplementedError("Genetic Operator must doing something with target")
