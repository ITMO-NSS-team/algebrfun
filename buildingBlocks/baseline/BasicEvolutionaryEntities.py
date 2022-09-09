"""
Содержит основные сущности, дополняющие функционал базовых структур и являющиеся основой для реализаций сущностей
эволюционного алгоритма.
"""

from copy import copy, deepcopy
from itertools import product
from typing import Union
from functools import reduce

import numpy as np
# from numba import njit
# import warnings
# warnings.filterwarnings("error")

from buildingBlocks.Globals.GlobalEntities import get_full_constant
import buildingBlocks.baseline.BasicStructures as Bs
from buildingBlocks.baseline.ParallelTools import map_wrapper, create_pool
import buildingBlocks.Globals.GlobalEntities as Bg
from buildingBlocks.supplementary.Other import mape

# @njit
# def fast_work(in_data, combin_params_flat, data):
#     current_arr = 0
#     dims = data.ndim
#     for i in range(in_data.shape[0]):
#         if i == 0:
#             current_arr = np.tensordot(combin_params_flat, in_data[i], axes=0)
#             continue
#         current_arr += np.tensordot(combin_params_flat, in_data[i], axes=0)  
    
#     res = []

#     for i, arr in enumerate(current_arr):
#         print('shp', i, arr.shape, data.shape)
#         current_arr[i] = np.exp(current_arr[i])
#         if not i:
#             res.append(np.tensordot(current_arr[i], data, dims))
#             continue
#         temp = arr.copy()
#         for j in range(i):
#             current_arr[i] -= (np.tensordot(temp, current_arr[j], dims) / np.tensordot(current_arr[j], current_arr[j], dims) * current_arr[j])
#         res.append(np.tensordot(current_arr[i], data, dims))

#     return res

class TerminalToken(Bs.Token):
    """
    TerminalToken is the token that returns a value as a vector whose evaluating
    requaires only numeric parameters.

    """
    def __init__(self, number_params: int = 0, params_description: dict = None, params: np.ndarray = None,
                 fixator: dict = None,
                 val: np.ndarray = None,
                 type_: str = 'TerminalToken', name_: str = None,
                 mandatory: float = 0,
                 optimize_id: int = None,
                 ):
        """

        Parameters
        ----------
        number_params: int
            Number of numeric parameters describing the behavior of the token.
        params_description: dict
            The dictionary of dictionaries for describing numeric parameters of the token.
            Must have the form like:
            {
                parameter_index: dict(name='name', bounds=(min_value, max_value)[, ...]),
                ...
            }
        params: numpy.ndarray
            Numeric parameters of the token for calculating its value.
        cache_val: bool
            If true, token value will be calculated only when its params are changed. Calculated value
            is written to the token property 'self.val'.
        fix_val: bool
            Defined by parameter 'cache_val'. If true, token value returns 'self.val'.
        fix: bool
            If true, numeric parameters will not be changed by optimization procedures.
        val: np.ndarray
            Value of the token.
        type_: str
        optimizer: str
        name_: str
            The name of the token that will be used for visualisation results and  some comparison operations.
        mandatory: float
            Unique id for the token. If not zero, the token must be present in the result construct.
        optimize_id: int
            Used for identifications by optimizers which token to optimize.
        """
        # todo кэш и вал отвечают за кэширование значения токена. Селф служит флагом его оптимизированности.
        if fixator is None:
            fixator = {}
        add_fixator = dict(cache=True, val=False, self=False)
        for key, value in add_fixator.items():
            if key not in fixator.keys():
                fixator[key] = value
        self.fixator = fixator
        
        self.val = val
        self.type = type_
        self.mandatory = mandatory
        self.optimize_id = optimize_id
        super().__init__(number_params=number_params, params_description=params_description,
                         params=params, name_=name_)

        self.variable_params = None


    def __select_parametrs__(self, in_data, data, population_size, gen=True):
        if self._number_params == 1:
            return
        sz = population_size
        answer = []
        for key_param in range(1, self._number_params):
            bounds = list(self.params_description[key_param]['bounds'])
            if bounds[1] == float('inf'):
                bounds[1] = sz
            params_lin = np.linspace(bounds[0], bounds[1], sz)
            params_wvar = np.tensordot(params_lin, in_data, axes=0)   
            params_wvar = (params_wvar - np.min(params_wvar))/(np.max(params_wvar) - np.min(params_wvar))
            params_wvar = np.exp(params_wvar)
            params_wvar = np.array([params_wvar[:, i, :] for i in range(in_data.shape[0])])
            all_combin = np.array([reduce(lambda x,y: x * y, el) for el in list(product(*params_wvar))])

            amplitudes = []
            for k, exp_indata_iter in enumerate(all_combin):
                amplitudes.append(np.tensordot(exp_indata_iter, data, data.ndim))
            shp = tuple([sz for _ in range(in_data.shape[0])])
            my_idxs = [ix for ix in np.ndindex(shp)]
            sort_ampls = np.array(sorted(list(zip(my_idxs, amplitudes)), key=lambda x: x[1], reverse=True))[:, 0]
            sort_ampls = np.array([np.array(el) for el in sort_ampls])
            for number_variable in range(in_data.shape[0]):
                try:
                    answer[number_variable].append(params_lin[sort_ampls[:, number_variable]])
                    # answer[number_variable].append(sort_ampls[:, number_variable])
                except:
                    answer.append([])
                    answer[number_variable].append(params_lin[sort_ampls[:, number_variable]])
                    # answer[number_variable].append(sort_ampls[:, number_variable])
        answer = np.array(answer)
        # self.variable_params = answer.reshape((in_data.shape[0], sz, self._number_params - 1))
        self.variable_params = answer.reshape((answer.shape[0], answer.shape[2], answer.shape[1]))
        

            

    def __getstate__(self):
        for key in self.__dict__.keys():
            if key in ('val',):
                self.__dict__[key] = None
            if key in ('forms',):
                self.__dict__[key] = []
        return self.__dict__

    def __setstate__(self, state: dict):
        self.__dict__ = state

    # def __eq__(self, other):
    #     if type(self) != type(other):
    #         return False
    #     are_parameters_eq = True
    #     are_parameters_compared = False
    #     for key in range(self._number_params):
    #         try:
    #             are_parameters_eq *= (mape(self.param(idx=key), other.param(idx=key))
    #                                   < self.params_description[key]['eq'])
    #             are_parameters_compared = True
    #         # if descriptor eq not exist
    #         except KeyError:
    #             continue
    #         # if descriptor eq = None
    #         except TypeError:
    #             continue
    #     return are_parameters_eq * are_parameters_compared


    def copy(self):
        new_copy = deepcopy(self)
        return new_copy

    def clean_copy(self):
        constants = get_full_constant()
        tmp_val = self.val
        self.val = None
        new_copy = deepcopy(self)
        self.val = tmp_val
        new_copy.params = np.zeros((len(constants['shape_grid']), new_copy._number_params))
        new_copy.fixator['self'] = False
        return new_copy

    def extra_clean_copy(self):
        new_copy = type(self)()
        new_copy.mandatory = self.mandatory * np.random.uniform(0.1, 2)
        new_copy.optimize_id = self.optimize_id
        return new_copy

    # Methods for work with params and its descriptions

    def check_params_description(self):
        """
        Check params_description for requirements for current token.
        """
        super().check_params_description()
        recomendations = "\nUse methods 'params_description.setter' or 'set_descriptor' to change params_descriptions"
        for key, value in self._params_description.items():
            assert 'bounds' in value.keys(), "Key 'bounds' must be in the nested " \
                                             "dictionary for each parameter" + recomendations
            assert (len(value['bounds']) == 2 and
                   np.all(value['bounds'][0] <= value['bounds'][1])), "Bounds of each parameter must have" \
                                                              " length = 2 and contain value" \
                                                              " boundaries MIN <= MAX." + recomendations

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params: np.ndarray):
        # self._params = np.array(params, dtype=float)
        self._params = params
        if len(params.shape) == 1:
            self._params = np.array([params])
        # потенциальные неожиданные баги от обрезания параметров
        self.check_params()
        self.fixator['val'] = False

    def check_params(self):
        super().check_params()
        for key, value in self._params_description.items():
            try:
                if self._params_description[key]['check']:
                    for i in range(self._params.shape[0]):
                        min_val, max_val = value['bounds']
                        try:
                            self._params[i][key] = min(self._params[i][key], min(max_val))
                        except:
                            self._params[i][key] = min(self._params[i][key], max_val)
                        try:
                            self._params[i][key] = max(self._params[i][key], max(min_val))
                        except:
                            self._params[i][key] = max(self._params[i][key], min_val)
            except KeyError:
                continue
            except ZeroDivisionError:
                continue

    def set_param(self, param, name=None, idx=None):
        super().set_param(param=param, name=name, idx=idx)
        self.fixator['val'] = False

    def init_params(self):
        try:
            for key, value in self._params_description.items():
                self.set_param(np.random.uniform(value['bounds'][0], value['bounds'][1]), idx=key)
        except OverflowError:
            raise OverflowError('Bounds have incorrect/infinite values')

    def value(self, grid: np.ndarray) -> np.ndarray:
        """
        Returns value of the token on the grid.
        Returns either cache result in self.val or calculated value in self.val by method self.evaluate().

        Parameters
        ----------
        grid: np.ndarray
            Grid for evaluation.

        Returns
        -------
        Value of the token.
        """
        if not self.fixator['val'] or self.val is None or self.val.shape[0] != grid.shape[-1]:
            self.val = self.evaluate(self.params, grid)
            self.fixator['val'] = self.fixator['cache']

            # эта централизация в целом то полезна (для ЛАССО например), но искажает продукт-токен
            # centralization
            # self.val -= np.mean(self.val)
        assert self.val.shape[0] == grid.shape[-1], "Value must be the same shape as grid "
        return self.val

    @staticmethod
    def evaluate(params: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """
        Calculating token value on the grid depending on parameters.
        Must be override/implement in each TerminalToken.
        May be not staticmethod if it is necessary.

        Parameters
        ----------
        params: numpy.ndarray
            Numeric token parameters.
        grid: numpy.ndarray
            Grid for evaluation.

        Returns
        -------
        numpy.ndarray
        """
        return np.zeros(grid.shape)

    def func_params(self, params, grid):
        params = list(params)
        for idx in range(len(params)):
            try:
                func = self.params_description[idx]['func']
                if func is not None:
                    params[idx] = func(params[idx], grid)
            except KeyError:
                continue
        return params


class ComplexToken(TerminalToken, Bs.ComplexStructure):
    """
    ComplexToken is the Token which consists other tokens (as subtokens in property self.subtokens)
    in addition to the numeric parameters.
    Example: Product of TerminalTokens.
    """
    def __init__(self, number_params: int = 0, params_description: dict = None, params: np.ndarray = None,
                 fixator: dict = None,
                 val: np.ndarray = None,
                 type_: str = 'TerminalToken', name_: str = None,
                 mandatory: float = 0,
                 optimize_id: int = None,
                 structure: list = None):
        """

        Parameters
        ----------
        See documentation TerminalToken.__init__.__doc__

        subtokens: list
            List of other tokens which current token uses for calculating its value.
        """

        super().__init__(number_params=number_params, params_description=params_description, params=params,
                         fixator=fixator, val=val, type_=type_,
                         name_=name_, mandatory=mandatory, optimize_id=optimize_id)

        # self._init_structure(structure)
        Bs.ComplexStructure.__init__(self, structure=structure)


def _methods_decorator(method):
    def wrapper(*args, **kwargs):
        self = args[0]
        self.change_all_fixes(False)
        return method(*args, **kwargs)
    return wrapper


class Individ(Bs.ComplexStructure):
    """
    Abstract class.
    Inidivid is a individual in population in the context of evolutionary algorithm.
    This class implements basic necessary functionality like work with chromosomes (whose consist of tokens as gens),
    with fitness of the individual and with influence of genetic operators that changes individual chromosome.
    """
    def __init__(self, structure: list = None,
                 fixator: dict = None,
                 fitness: float = None):
        """

        Parameters
        ----------
        genetic_operators: dict
            Set of names of genetic operators that can influence the individ.
            These names are mapped to the implementations of the operators in the OperatorsKeeper object.
            Form example:
            {
                'operator name for Individ': 'operator name in OperatorsKeeper',
                ...
            }
        chromo: list
            List of tokens as gens.
        fitness: float
            Numeric metric of the Individ fitness
        store: dict
            Caching influence of genetic operators.
        """
        if fixator is None:
            fixator = {}
        self.fixator = fixator

        super().__init__(structure)

        self.fitness = fitness

    @property
    def structure(self) -> list:
        return self._structure

    @structure.setter
    def structure(self, structure: list) -> None:
        self.change_all_fixes(False)
        assert type(structure) == list, "structure must be a list"
        self._structure = structure

    @_methods_decorator
    def set_substructure(self, substructure, idx: int) -> None:
        super().set_substructure(substructure, idx)

    @_methods_decorator
    def get_substructure(self, idx: int):
        super().get_substructure(idx)

    @_methods_decorator
    def add_substructure(self, substructure, idx: int = -1) -> None:
        super().add_substructure(substructure, idx)

    @_methods_decorator
    def del_substructure(self, substructure):
        super().del_substructure(substructure)

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, fitness):
        self._fitness = fitness

    def change_all_fixes(self, value: bool = False):
        for key in self.fixator.keys():
            self.fixator[key] = value

    def copy(self):
        new_copy = deepcopy(self)
        return new_copy

    def apply_operator(self, name: str, *args, **kwargs):
        """
        Apply an operator with the given name.

        Parameters
        ----------
        name: str
            Name of the operator in genetic_operators dict.

        args
        kwargs

        Returns
        -------
        None

        Args:
            keeper:
            keeper:
        """
        operators = Bg.get_operators()
        try:
            operator = operators[name]
        except KeyError:
            raise KeyError("Operator with name '{}' is not implemented in"
                           " object {}".format(name, operators))
        except TypeError:
            raise TypeError("Argument 'operators' cannot be '{}'".format(type(operators)))
        return operator.apply_to(self, *args, **kwargs)


class Population(Bs.ComplexStructure):
    """
    Abstract class (methods 'evolutionary' and 'evolutionary_step' must be implemented).
    Population contains set of individuals.
    This class implements basic necessary functionality like work with population and with genetic operators that
    change population (add new individs or change existing ones).
    Methods
    """
    def __init__(self, structure: list = None):
        """

        Parameters
        ----------
        genetic_operators: dict
            Set of names of genetic operators that can influence population (individuals in population).
            These names are mapped to the implementations of the operators in the OperatorsKeeper object.
            Form example:
            {
                'operator name for Individ': 'operator name in OperatorsKeeper',
                ...
            }
        population: list
            List of individuals in population.
        """
        super().__init__(structure=structure)

    def apply_operator(self, name: str, *args, **kwargs):
        operators = Bg.get_operators()
        try:
            operator = operators[name]
        except KeyError:
            raise KeyError("Operator with name '{}' is not implemented in"
                           " object {}".format(name, operators))
        except TypeError:
            raise TypeError("Argument 'operators' cannot be '{}'".format(type(operators)))
        return operator.apply_to(self, *args, *kwargs)

    def evolutionary(self):
        """
        Evolutionary process of the population.
        """
        raise NotImplementedError("Define evolution by 'Population.evolutionary_step()'")


class GeneticOperatorIndivid(Bs.GeneticOperator):
    """
    Genetic Operator influencing object Individ.
    Change Individ, doesn't create a new one.
    """
    def __init__(self, params: dict = None):
        super().__init__(params=params)

    def apply(self, individ, *args, **kwargs) -> None:
        raise NotImplementedError("Genetic Operator must doing something with Individ/Population")

    def apply_to(self, individ, *args, **kwargs) -> None:
        """Использует метод apply, не переопределять в наследниках."""
        if kwargs:
            tmp_params = {}
            for key, value in self.params.items():
                tmp_params[key] = value
            for key, value in kwargs.items():
                if key in self.params.keys():
                    self.params[key] = kwargs[key]
        else:
            tmp_params = self.params

        ret = self.apply(individ, *args, **kwargs)
        self.params = tmp_params
        return ret


class GeneticOperatorPopulation(Bs.GeneticOperator):  
    """
    Genetic Operator influencing list of Individs in Population.
    May be parallelized.
    May change Individs in population and create new ones. Return new list of Individs.
    """
    def __init__(self, params: dict = None):
        super().__init__(params=params)
        if 'parallelise' not in self.params.keys():
            self.params['parallelise'] = False

    def apply(self, population: Population, *args, **kwargs) -> Union[None, Population]:
        raise NotImplementedError("Genetic Operator must doing something with Individ/Population")

    def apply_to(self, population: Population, *args, **kwargs) -> Union[None, Population]:
        if self.params['parallelise']:
            # for individ in population.structure:
            #     individ.send_preparing()
            create_pool()
            return map_wrapper(type(self).apply, self, population=population)  #todo разобраться с передачей аргументов
        return self.apply(population, *args, **kwargs)


class DifferentialToken(Bs.Token):
    def __init__(self, number_params: int = 0, params_description: dict = None, params: np.ndarray = None, name_: str = None):
        super().__init__(number_params, params_description, params, name_)
