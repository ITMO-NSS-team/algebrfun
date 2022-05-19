"""
Contains inheritors/implementations of baseline classes for Token.
Любая параметрическая функция выражается в виде токена и реализует его основные методы.
Можно писать свои токены, и реализуя их API, алгоритм будет брать их в обработку.
Узкое место для ускорения numba.
"""
from copy import deepcopy
# from buildingBlocks.baseline.Tokens import TerminalToken, ComplexToken
from buildingBlocks.Globals.GlobalEntities import get_full_constant
from buildingBlocks.baseline.BasicEvolutionaryEntities import TerminalToken, ComplexToken
import numpy as np
from functools import reduce
from itertools import product

from buildingBlocks.supplementary.Other import mape


# from numba import njit


class Constant(TerminalToken):
    def __init__(self, number_params=1,
                 params_description=None,
                 params=np.array([1.]), val=None,
                 name_=None, mandatory=0):
        if params_description is None:
            params_description = {
                0: dict(name='Amplitude', bounds=(-1., 1.)),
            }
        super().__init__(number_params=number_params, params_description=params_description,
                         params=params, val=val, name_=name_, mandatory=mandatory)
        self.type = 'Constant'
        # self.init_val = deepcopy(val)
        # self.init_val = name_
        self.fixator['self'] = True
    # TODO Проверить теорию о влиянии замены значений инит вал на имя в общедоступном словаре констант на скорость \\

    def clean_copy(self):
        new_copy = super().clean_copy()
        if self.mandatory != 0:
            new_copy.set_param(1., name='Amplitude')
            new_copy.fixator['self'] = True
        return new_copy

    def evaluate(self, params, grid):
        constants = get_full_constant()
        # !!! очень важный минус, который инвертирует таргет для его компенсации суммой других токенов
        return -params[0] * constants[self.name_]

    def name(self, with_params=False):
        return '{}{}'.format(round(self.params[0][0], 3), self.name_)

    def __eq__(self, other):
        if type(self).__name__ == type(other).__name__:
            return self.name_ == other.name_
        return False


class Power(TerminalToken):
    def __init__(self, number_params=2, params_description=None,
                 params=None, name_='Power', optimize_id=None):
        if params_description is None:
            params_description = {
                0: dict(name='Amplitude', bounds=(-1., 1.)),
                1: dict(name='Power', bounds=(0., 3.), check=True)
            }
        super().__init__(number_params=number_params, params_description=params_description,
                         params=params, name_=name_, optimize_id=optimize_id)
        self.type = "NonPeriodic"

    def each_evaluate(self, params, t):
        params = self.func_params(params, t)
        return t ** params[1]
    
    def evaluate(self, params, t):
        result = np.nan
        for i in range(t.shape[0]):
            cur_temp = self.each_evaluate(params[i], t[i])
            if np.isnan(result):
                result = cur_temp
            else:
                result += cur_temp
        return params[0][0] * result

    def name(self, with_params=False):
        # return self.name_ + str(self.params)
        a, n = self.params
        return '{}(t**{})'.format(round(a, 2), round(n, 2))

    def __eq__(self, other):
        if type(self) == type(other):
            if not self.fixator['self'] or not other.fixator['self']:
                return False
            if mape(self.params[1], other.params[1]) < 0.1:
                return True
            return False
        return False


class Sin(TerminalToken):
    def __init__(self, number_params=3, params_description=None,
                 params=None, name_='Sin', optimize_id=None):
        if params_description is None:
            params_description = {
                0: dict(name='Amplitude', bounds=(0., 1.)),
                1: dict(name='Frequency', bounds=(0., float('inf'))),
                2: dict(name='Phase', bounds=(0., 1.))
            }
        super().__init__(number_params=number_params, params_description=params_description,
                         params=params, name_=name_, optimize_id=optimize_id)
        self.type = "Periodic"

    def each_evaluate(self, params, t):
        # todo maybe problems
        params = np.abs(params)
        params = self.func_params(params, t)
        # if (params[0:2] == 0).any():
        #     return np.zeros(t.shape)
        # return params[0] * np.sin(1 * np.pi * (2 * params[1] * t + abs(math.modf(params[2])[0])))
        return np.sin(2 * np.pi * (params[1] * t + params[2]))

    def evaluate(self, params, t):
        result = np.nan
        if len(params.shape) == 1:
            params = list([params])
        # print("prm", params)
        for i in range(t.shape[0]):
            # print("allany", params[i], t[i])
            cur_temp = self.each_evaluate(params[i], t[i])
            # print("allanyres", cur_temp)
            if np.all(np.isnan(result)):
                result = cur_temp
            else:    
                result *= cur_temp
        return params[0][0] * result


    def name(self, with_params=False):
        # return self.name_ + str(self.params)
        a, w, fi = self.params[0] # !!!!!
        return '{}Sin({}t + {}pi)'.format(round(a, 2), round(w, 2), round(fi, 2))

    def __eq__(self, other):
        if type(self) == type(other):
            if not self.fixator['self'] or not other.fixator['self']:
                return False
            self_freq = self.param(name='Frequency')
            other_freq = other.param(name='Frequency')
            self_phase = self.param(name='Phase')
            other_phase = other.param(name='Phase')
            # if np.all(self.params == np.zeros(self.params.shape)) or \
            #         np.all(other.params == np.zeros(other.params.shape)):
            #     return False
            print("freqs", type(self_freq), type(other_freq), self_freq, other_freq)
            if np.all(self_freq == other_freq) & np.all(other_freq == np.zeros(len(other_freq))):
                return True
            if np.all(self_phase == other_phase) & np.all(other_phase == np.zeros(len(other_phase))):
                return np.all(abs((self_freq - other_freq) / (self_freq + other_freq)) < np.full(other_freq.shape, 0.01))
            return (np.all(abs((self_freq - other_freq) / (self_freq + other_freq)) < np.full(other_freq.shape, 0.01)) &
                    np.all(abs((self_phase - other_phase) / (self_phase + other_phase)) < np.full(other_freq.shape, 0.25)))
            # return abs((self_freq - other_freq) / (self_freq + other_freq)) < 0.05
            # return abs((self.param(name='Frequency') - other.param(name='Frequency')) /
            #            (self.param(name='Frequency') + other.param(name='Frequency'))) < 0.05
        return False


class ImpSingle(TerminalToken):
    def __init__(self, number_params=6,
                 params_description=None, params=None):
        if params_description is None:
            params_description = {
                0: dict(name='Amplitude', bounds=(0., 1.)),
                1: dict(name='Pulse start', bounds=(0., float('inf'))),
                2: dict(name='Pulse front duration', bounds=(0., float('inf'))),
                3: dict(name='Pulse recession duration', bounds=(0., float('inf'))),
                4: dict(name='Front power', bounds=(0.05, 2.), check=True),
                5: dict(name='Recession power', bounds=(0.05, 2.), check=True)
            }
        super().__init__(number_params=number_params, params_description=params_description,
                         params=params)
        self.type = "NonPeriodic"

        # add new val to use in correlation and other
        self.non_zero_val = None

    # @staticmethod
    def each_evaluate(self, params, t):
        params[[0, 4, 5]] = np.abs(params[[0, 4, 5]])

        A, T1, T2, T3, p1, p2 = params

        cond1 = (t >= T1) & (t < T1 + T2)
        cond2 = (t >= T1 + T2) & (t <= (T1 + T2 + T3))
        m = np.zeros(len(t))
        if T2 != 0:
            m[cond1] = (np.abs(t[cond1] - T1) / T2) ** p1#np.abs(p1)
        if T3 != 0:
            m[cond2] = (np.abs(t[cond2] - (T1 + T2 + T3)) / T3) ** p2#np.abs(p2)
        # обрезаем во избежание экспоненциального затухания к нулю (плохо при оптимизации)
        m[np.abs(m) < 0.02] = 0
        return m

    def evaluate(self, params, t):
        result = np.nan
        for i in range(t.shape[0]):
            cur_val = self.each_evaluate(params[i], t[i])
            if np.isnan(result):
                result = cur_val
            else:
                result *= cur_val
        
        return params[0][0] * result



class ImpSingle2(TerminalToken):
    def __init__(self, number_params=6,
                 params_description=None, params=None):
        if params_description is None:
            params_description = {
                0: dict(name='Amplitude', bounds=(0., 1.)),
                1: dict(name='Pulse start', bounds=(0., float('inf'))),
                2: dict(name='Duration', bounds=(0., float('inf'))),
                3: dict(name='Ratio',  bounds=(0.05, 0.98)),
                4: dict(name='Front power', bounds=(0.05, 2.), check=True),
                5: dict(name='Recession power', bounds=(0.05, 2.), check=True)
            }
        super().__init__(number_params=number_params, params_description=params_description,
                         params=params)
        self.type = "NonPeriodic"

        # add new val to use in correlation and other
        self.non_zero_val = None

    @staticmethod
    def each_evaluate(params, t):
        params[[0, 4, 5]] = np.abs(params[[0, 4, 5]])
        A, T1, T, ratio, p1, p2 = params

        T2 = T*ratio
        T3 = T - T2

        cond1 = (t >= T1) & (t < T1 + T2)
        cond2 = (t >= T1 + T2) & (t <= (T1 + T2 + T3))
        m = np.zeros(len(t))
        if T2 != 0:
            m[cond1] = (np.abs(t[cond1] - T1) / T2) ** p1#np.abs(p1)
        if T3 != 0:
            m[cond2] = (np.abs(t[cond2] - (T1 + T2 + T3)) / T3) ** p2#np.abs(p2)
        #!!!
        m[np.abs(m) < 0.02] = 0
        return m

    
    def evaluate(self, params, t):
        result = np.nan
        if len(t.shape) == 1:
            t = np.array([t]) # !!!!!!
        for i in range(t.shape[0]):
            cur_val = self.each_evaluate(params[i], t[i])
            if np.isnan(result):
                result = cur_val
            else:
                result *= cur_val

        return params[0][0] * result

class Imp(TerminalToken):
    def __init__(self, number_params=7, params_description=None, params=None, name_='Imp', optimize_id=None):
        if params_description is None:
            params_description = {
                0: dict(name='Amplitude', bounds=(0., 1.)),
                1: dict(name='Frequency', bounds=(0., float('inf'))),
                2: dict(name='Zero part of period', bounds=(0.05, 0.98)),
                3: dict(name='Front part of pulse duration', bounds=(0.05, 0.98)),
                4: dict(name='Front power', bounds=(0.05, 2.), check=True),
                5: dict(name='Recession power', bounds=(0.05, 2.), check=True),
                6: dict(name='Phase', bounds=(0., 1.))
            }
        super().__init__(number_params=number_params, params_description=params_description,
                         params=params, name_=name_, optimize_id=optimize_id)
        self.type = "Periodic"

    def each_evaluate(self, params, t):
        params = np.abs(params)
        if (params[0:2] == 0).any():
            return np.zeros(t.shape)
        params = self.func_params(params, t)
        A = params[0]
        fi = params[-1]

        T = 1 / params[1]
        n1 = np.abs(params[2])
        T1 = n1 * T
        n2 = np.abs(params[3])
        T2 = n2 * (T - T1)
        T3 = T - T1 - T2

        p1 = params[4]
        p2 = params[5]

        # !!! невероятно важно учитывать что фаза умножается на период, много где это учитывается (ImpC)
        t1 = (t + fi * T) % T

        cond1 = (t1 >= T1) & (t1 < T1 + T2)
        cond2 = (t1 >= T1 + T2) & (t1 <= T)
        try:
            m = np.zeros(len(t))
            if T2 != 0:
                m[cond1] = (np.abs(t1[cond1] - T1) / T2) ** p1#np.abs(p1)
            if T3 != 0:
                m[cond2] = (np.abs(t1[cond2] - T) / T3) ** p2#np.abs(p2)
        except:
            m = None
            if cond1:
                m = np.abs((t1 - T1) / T2) ** np.abs(p1)
            elif cond2:
                m = np.abs((t1 - T) / T3) ** np.abs(p2)
        m[np.abs(m) < 0.02] = 0
        return m

    def evaluate(self, params, t):
        result = np.nan
        if len(params.shape) == 1:
            params = list([params])
        for i in range(t.shape[0]):
            cur_val = self.each_evaluate(params[i], t[i])
            if np.all(np.isnan(result)):
                result = cur_val
            else:
                result *= cur_val
        
        return params[0][0] * result

    def name(self, with_params=False):
        # return self.name_ + str(self.params)
        a, w, fi = self.params[0][[0, 1, -1]] # !!!!!
        return '{}Imp({}t + {}pi)'.format(round(a, 2), round(w, 2), round(fi, 2))

    # TODO придумать нормальный метод сравнения
    # def __eq__(self, other):
    #     if isinstance(self, type(other)):
    #         self_freq = self.param(name='Frequency')
    #         other_freq = other.param(name='Frequency')
    #         self_phase = self.param(name='Phase')
    #         other_phase = other.param(name='Phase')
    #         if self_freq == other_freq == 0:
    #             return True
    #         if self_phase == other_phase == 0:
    #             return mape(self_freq, other_freq) < 0.05
    #         return (mape(self_freq, other_freq) < 0.05 and
    #                 mape(self_phase, other_phase) < 0.25)
    #         # return abs((self.param(name='Frequency') - other.param(name='Frequency')) /
    #         #            (self.param(name='Frequency') + other.param(name='Frequency'))) < 0.05
    #     return False


class Product(ComplexToken):
    """Сложный токен, реализующий произведение нескольких простых (в потенциале и сложных) токенов"""

    def __init__(self, number_params=2,
                 params_description=None,
                 params=np.array([1., 3]),
                 val=None,
                 name_=None,
                 structure=None):
        if params_description is None:
            params_description = {
                0: dict(name='Amplitude', bounds=(-1., 1.)),
                1: dict(name='max_subtokens_len', bounds=(2, float('inf')))
            }
        super().__init__(number_params=number_params, params_description=params_description,
                         params=params, val=val, name_=name_, structure=structure)

        self.type = 'Product'

        func_mas = [self.set_substructure,  self.add_substructure, self.del_substructure]
        for idx in range(len(func_mas)):
            func_mas[idx] = self._methods_wrapper(func_mas[idx])

        self._check_mandatory()

    def _check_mandatory(self):
        """
        If some subtoken in ComplexToken is mandatory then ComplexToken is mandatory too.

        Returns
        -------

        """
        for token in self.structure:
            if token.mandatory != 0:
                self.mandatory = np.random.uniform()
                return
        self.mandatory = 0

    @staticmethod
    def _set_substructure_amplitude(substructure, amplitude=1.):
        try:
            for token in substructure:
                token.set_param(amplitude, name='Amplitude')
        except:
            substructure.set_param(amplitude, name='Amplitude')

    @staticmethod
    def _methods_wrapper(method):
        def wrapper(*args, **kwargs):
            self = args[0]
            substructure = args[1]
            self._set_substructure_amplitude(substructure)
            res = method(*args, **kwargs)
            self.fixator['val'] = False
            self._check_mandatory()
            return res
        return wrapper

    # todo Все махинации с манипулироваинем амплитудами сделаны ради ПродуктТокена, а нужен общий функционал
    @property
    def structure(self) -> list:
        return self._structure

    # @_methods_wrapper
    @structure.setter
    def structure(self, structure):
        self._set_substructure_amplitude(structure)
        assert type(structure) == list, "structure must be a list"
        self._structure = structure
        self.fixator['val'] = False
        self._check_mandatory()

    def add_substructure(self, substructure):
        if len(self.structure) >= int(self.params[1]):
            return self.set_substructure(substructure, idx=np.random.randint(0, len(self.structure)))

        self.fixator['val'] = False
        self.structure.append(substructure)
        return

    def each_evaluate(self, params, grid):
        # self._fix_val = reduce(lambda val, x: val*x,
        #                        list(map(lambda x: x._fix_val, self.subtokens)))
        return reduce(lambda val, x: val * x,
                                  list(map(lambda x: x.value(grid), self.structure)))
    
    def evaluate(self, params, t):
        result = np.nan
        for i in range(t.shape[0]):
            cur_val = self.each_evaluate(params[i], t[i])
            if np.isnan(result):
                result = cur_val
            else:
                result *= cur_val
        
        return params[0][0] * result

    def name(self, with_params=False):
        # if self.name_ is not None:
        #     return str(self.params[0]) + self.name_
        s = '('
        for i in self.structure:
            s += i.name_
        s += ')'
        # if with_params:
        #     return type(self).__name__ + s
        return str(self.params[0]) + type(self).__name__ + s

    def __eq__(self, other):
        if type(self).__name__ == type(other).__name__:
            if len(self.structure) == len(other.subtokens):
                for token in self.structure:
                    if self.structure.count(token) != other.subtokens.count(token):
                        return False
                return True
        return False


# тип соответствия и количество используемых временных параметров
imp_relations = {
    Imp: [ImpSingle, 3],
}


class ImpComplex(ComplexToken):
    """Самый сложный токен (страдания). Состоит из последовательно идущих одиночных импульсов с разными параметрами"""
    def __init__(self, pattern: Imp = None,
                 number_params=1, params_description=None, params=np.array([1.]),
                 val=None,
                 type_: str = 'ComplexToken', name_: str = 'ImpComplex',
                 optimize_id=None,
                 mandatory=0,
                 structure: list = None,
                 single_imp=ImpSingle2()):
        self.pattern = deepcopy(pattern)
        self.single_imp = single_imp
        if params_description is None:
            params_description = {
                0: dict(name='Amplitude', bounds=(-1., 1.))
            }
        super().__init__(number_params=number_params, params_description=params_description, params=params,
                         val=val,
                         type_=type_, name_=name_,
                         optimize_id=optimize_id,
                         mandatory=mandatory,
                         structure=structure)

    def extra_clean_copy(self):
        new_copy = type(self)(pattern=self.pattern.copy())
        new_copy.mandatory = self.mandatory * np.random.uniform(0.1, 2)
        new_copy.optimize_id = self.optimize_id
        return new_copy

    def name(self, with_params=False):
        return '{}ImpComplex({}t)'.format(round(self.params[0][0], 3), round(self.pattern.params[0][1], 3))

    def init_structure_from_pattern(self, grid):
        if len(self.structure) != 0:
            return
        dif_params = deepcopy(self.pattern.params)
        # print(self, params)
        new_imps = [[] for _ in range(dif_params.shape[0])]
        for idx, params in enumerate(dif_params):
            A, w, n1, n2, p1, p2, fi = params
            T = 1/w
            T1 = T*n1
            T2 = (T - T1)*n2
            T3 = T - T2 - T1
            grid_max = grid[idx].max()

            pulse_start = -fi * T + T1
            while pulse_start < grid_max: # что за условие почему именно такое
                new_params = np.array([A, pulse_start, T2+T3, n2, p1, p2])
                # new_imp = type(self.single_imp)(params=np.array([new_params]))
                # self.add_substructure(new_imp)
                new_imps[idx].append(new_params)
                pulse_start += T
        for combin in list(product(*new_imps)):
            new_imp = type(self.single_imp)(params=np.array(combin))
            self.add_substructure(new_imp)


    def fix_structure(self, flag=True):
        for token in self.structure:
            token.fixator['val'] = flag

    def value(self, grid):
        # fixed_optimized_tokens_in_structure = list(filter(lambda token: token.fixator['self'],
        #                                                   self.structure))
        fixed_optimized_tokens_in_structure = self.structure
        if len(fixed_optimized_tokens_in_structure) == 0:
            return np.zeros(grid.shape)
        ampl = self.param(name='Amplitude') #!!! может быть очень опасным багом если параметр связан с паттерном
        res = reduce(lambda x, y: x + y,
                     list(map(lambda token: token.value(grid),
                              fixed_optimized_tokens_in_structure)))
        return ampl*res

    def init_params(self):
        self.pattern.init_params()
        self.params = deepcopy(self.pattern.params)

    # todo вынести эту функциональность
    # def make_properties(self):
    #     mas = []
    #     for i in range(len(self.structure[0].params)):
    #         ps = np.array(list(map(lambda x: x.params[i], self.structure)))
    #         mas.append(ps)
    #     mas[1] = np.sort(mas[1])
    #     mas[1] = np.abs(mas[1][1:] - mas[1][:-1])
    #     return mas

    # def get_imps_bounds(self, t):
    #     imps_bounds = deepcopy(self.pattern.bounds[:-1])
    #     imps_bounds[1] = (0, t.max())
    #     imps_bounds[2] = (0, imps_bounds[1][1] / 4)
    #     imps_bounds[3] = (0, imps_bounds[1][1] / 4)
    #     return imps_bounds

    # def value(self, grid):
    #     if self.value_type == 'norm':
    #         self.init_imps(grid)
    #         return self.value_imps(grid)
    #     else:
    #         return self.sample_value(grid)
