from copy import deepcopy
import numpy as np
from functools import reduce
from itertools import product
from scipy.optimize import minimize
# from buildingBlocks.Globals.GlobalEntities import get_full_constant

class Token:
    """
    Basic function as component of expression's structure.

    Attributes
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
    owner_id:
        Used for identifications by individ which that token is belongs.

    Methods
    -------

    """

    def __init__(self, number_params: int=0, params_description: dict={}, params: np.ndarray=None,
                 val: np.ndarray=None, type_: str="Token", name_: str=None,
                 optimize_id: int=None) -> None:

        self.val = val
        self.type = type_
        self.optimize_id = optimize_id
        self._number_params = number_params
        self._params_description = params_description
        # self.variable_params = [[0]]
        
        if params is None:
            self.params = np.array([np.zeros(self._number_params)])
        else:
            self.params = np.array(params, dtype=object)

        if name_ is None:
            name_ = type(self).__name__
        self.name_ = name_

    
    
    def __eq__(self, other: object) -> bool:
        assert isinstance(self, Token), "Objects are different types"
        assert isinstance(other, Token), "Objects are different types"
        return self.name_ == other.name_ and np.allclose(self.params[:, 1:], other.params[:, 1:])

    def __getstate__(self):
        for key in self.__dict__.keys():
            if key in ('val',):
                self.__dict__[key] = None
            if key in ('forms',):
                self.__dict__[key] = []
        return self.__dict__

    def __setstate__(self, state: dict):
        self.__dict__ = state

    

    def name(self):
        return self.name_

    def value(self, grid: np.ndarray):
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
        # if not self.fixator['val'] or self.val is None or self.val.shape[0] != grid.shape[-1]:
        self.val = self.evaluate(self.params, grid)
        # self.fixator['val'] = self.fixator['cache']

            # эта централизация в целом то полезна (для ЛАССО например), но искажает продукт-токен
            # centralization
            # self.val -= np.mean(self.val)
        assert self.val.shape[0] == grid.shape[-1] or self.val.shape[0] == 1, "Value must be the same shape as grid "
        return self.val

    def evaluate(self, params: np.ndarray, grid: np.ndarray) -> np.ndarray:
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

    def copy(self):
        new_copy = deepcopy(self)
        return new_copy

    def clean_copy(self):
        new_copy = deepcopy(self)
        new_copy.val = None
        new_copy.params = np.zeros((self.params.shape))
        # new_copy.fixator['self'] = False
        return new_copy

    
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
            assert 'bounds' in value.keys(), "Key 'bounds' must be in the nested " \
                                             "dictionary for each parameter" + recomendations
            assert (len(value['bounds']) == 2 and
                   np.all(value['bounds'][0] <= value['bounds'][1])), "Bounds of each parameter must have" \
                                                              " length = 2 and contain value" \
                                                              " boundaries MIN <= MAX." + recomendations

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
        if len(params.shape) == 1:
            self._params = np.array([params])
        # self._params = np.array([params])
        # потенциальные неожиданные баги от обрезания параметров
        self.check_params()
        # self.fixator['val'] = False

    def check_params(self):
        isinstance(self._params, np.ndarray)
        assert self._params.shape[-1] == self._number_params, "The number of parameters does not match the length of params array"\
                                                  + f"\nUse methods 'params.setter' or 'set_param' to change params: {self.name}, {self._params.shape[-1]}, {self._number_params}"
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
            try:
                for i in range(len(param)):
                    self._params[i][idx] = param[i] # разобраться, так как не все параметры расширяемы!!!!
            except:
                self._params[0][idx] = param
        except IndexError:
            raise IndexError('"{}" have no parameter with index "{}"'.format(self, idx))
        self.check_params()

            
    def _find_initial_approximation_(self, in_data, data, population_size, gen=True): # пока что уберу, для начала надо разобраться
        sz = population_size
        if self._number_params == 1:
            bounds = list(self.params_description[0]['bounds'])
            tmpl = np.linspace(bounds[0], bounds[1], sz)
            res = []
            for _ in range(in_data.shape[0]):
                res.append(tmpl)
            self.variable_params = np.array(res)
            self.variable_params= self.variable_params.reshape(self.variable_params.shape[0], sz, self._number_params)
            return
        answer = []
        for key_param in range(1, self._number_params):
            bounds = list(self.params_description[key_param]['bounds'])
            if bounds[1] == float('inf'):
                bounds[1] = sz
            params_lin = np.linspace(bounds[0], bounds[1], sz)
            params_wvar = self.preprocess_fft(in_data, params_lin)
            # params_wvar = np.tensordot(params_lin, in_data, axes=0)   
            # params_wvar = (params_wvar - np.min(params_wvar))/(np.max(params_wvar) - np.min(params_wvar))
            # params_wvar = np.exp(params_wvar)
            params_wvar = np.array([params_wvar[:, i, :] for i in range(in_data.shape[0])])
            all_combin = np.array([reduce(lambda x,y: x * y, el) for el in list(product(*params_wvar))])

            amplitudes = []
            for k, exp_indata_iter in enumerate(all_combin):
                # amplitudes.append(np.tensordot(exp_indata_iter, data.data, data.data.ndim))
                ampl = np.tensordot(exp_indata_iter, data.data, data.data.ndim)
                if not k:
                    amplitudes.append(ampl)
                    continue
                fix = exp_indata_iter
                for j in range(k):
                    all_combin[k] -= (np.sum(fix * all_combin[j]) / np.sum(all_combin[j] * all_combin[j]) * all_combin[j])
                amplitudes.append(np.tensordot(all_combin[k], data.data, data.data.ndim))
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

    def _select_params(self):
        shp = self.variable_params.shape
        index = np.random.choice(shp[-2])
        set_params = self.variable_params[..., index, :]
        ampl = np.ones(shp[0]).reshape(1, -1).T
        if self._number_params == 1:
            self.params = ampl
        else:
            self.params = np.hstack((ampl, set_params))
    
    def select_best_params(self, individ):
        shp = self.variable_params.shape
        flag = False
        last_params = self.params.copy()
        l_fitness = individ.fitness
        for i in range(shp[-2]):
            set_params = self.variable_params[..., i, :]
            ampl = np.ones(shp[0]).reshape(1, -1).T
            if self._number_params == 1:
                self.params = ampl
            else:
                self.params = np.hstack((ampl, set_params))

            individ.apply_operator("VarFitnessIndivid")
            if l_fitness < individ.fitness:
                self.params = last_params
            else:
                last_params = self.params.copy()
    
    def check_border(self, ind, b_params):
        borders_of_param = self.params_description[ind]['bounds']
        if borders_of_param[0] > b_params[0]:
            b_params[0] = borders_of_param[0]
        
        if not isinstance(borders_of_param[1], np.inf) and b_params[1] > borders_of_param[1]:
            b_params[1] = borders_of_param[1]
        
        return b_params


    @staticmethod
    def _fitness_wrapper(params, *args):
        individ, token, shp = args

        token.params = params.reshape(shp)

        individ.apply_operator("VarFitnessIndivid")
        return individ.fitness

    def find_params_(self, individ):
        # term.expression_token = self
        # individ.structure.append(term)

        self._select_params()
        shp = self.params.shape
        x0 = self.params

        res = minimize(self._fitness_wrapper, x0.reshape(-1), args=(individ, self, shp))

        self.params = res.x.reshape(shp)

    
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
    
