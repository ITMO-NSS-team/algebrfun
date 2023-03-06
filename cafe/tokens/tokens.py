import numpy as np

from .base import Token

class Constant(Token):
    """
    Basic function for decript 
    """
    def __init__(self, number_params: int = 1, params_description: dict = None, params: np.ndarray = None, val: np.ndarray = None, name: str = 'target', optimize_id: int = 2) -> None:
        if params_description is None:
            params_description = {
                0: dict(name='Amplitude', bounds=(-100, 100)),
            }
        super().__init__(number_params=number_params, params_description=params_description, params=params, 
                         val=val, name_=name, optimize_id=optimize_id)

        self.type = 'Constant'

    def __eq__(self, other: object) -> bool:
        assert isinstance(self, Token), "Objects are different types"
        assert isinstance(other, Token), "Objects are different types"

        ex_1 = self.name_ == other.name_
        # if not ex_1:
        #     return False
        # ex_0 = self._params == other._params

        return ex_1
        # return self.name_ == other.name_ and np.allclose(self.params, other.params)

    def evaluate(self, params, grid):
        # constants = get_full_constant()
        # !!! очень важный минус, который инвертирует таргет для его компенсации суммой других токенов
        # return -params[0] * constants[self.name_]
        
        # return self.params[0] * np.ones_like(constants[self.name_].data)
        return self.params[0]
    
    def name(self, with_params=False):
        return '{}{}'.format(round(self.params[0][0], 3), self.name_)


class Power(Token):
    def __init__(self, number_params=2, params_description=None,
                 params=None, name='Power', optimize_id=2):
        if params_description is None:
            params_description = {
                0: dict(name='Amplitude', bounds=(-1., 1.)),
                1: dict(name='Power', bounds=(0., 3.), check=True)
            }
        super().__init__(number_params=number_params, params_description=params_description,
                         params=params, name_=name, optimize_id=optimize_id)
        self.type = "NonPeriodic"
        
    def each_evaluate(self, params, t):
        params = self.func_params(params, t)
        return t ** params[1]
    
    def evaluate(self, params, t):
        result = np.nan
        for i in range(t.shape[0]):
            # print("beforrre", t[i])
            if len(t[i][t[i]<0]) > 0:
                params[i][1] = round(params[i][1])
            cur_temp = self.each_evaluate(params[i], t[i])
            if np.all(np.isnan(result)):
                result = cur_temp
            else:
                result += cur_temp
        self.params = params
        # print("checking power", result, params)
        return params[0][0] * result

    def name(self, with_params=False):
        # return self.name_ + str(self.params)
        ampl = self.params[0][0]
        # a, n = self.params
        str_result = '{}('
        for iter_params in self.params:
            a, n = iter_params
            str_result += 't**{} + '.format(round(n, 2))

        try:
            str_result = str_result[:-3] + ')'
        except:
            print("strange size")

        # str_result += ')'
        return str_result.format(round(ampl, 2))

class Sin(Token):
    def __init__(self, number_params=3, params_description=None,
                 params=None, name='Sin', optimize_id=1):
        if params_description is None:
            params_description = {
                0: dict(name='Amplitude', bounds=(0., 10.)),
                1: dict(name='Frequency', bounds=(0.95, 1.05)),
                2: dict(name='Phase', bounds=(0., 1.))
            }
        super().__init__(number_params=number_params, params_description=params_description,
                         params=params, name_=name, optimize_id=optimize_id)
        self.type = "Periodic"

    def each_evaluate(self, params, t):
        # todo maybe problems
        params = np.abs(params)
        params = self.func_params(params, t)
        # if (params[0:2] == 0).any():
        #     return np.zeros(t.shape)
        # return params[0] * np.sin(1 * np.pi * (2 * params[1] * t + abs(math.modf(params[2])[0])))
        # return np.sin(2 * np.pi * (params[1] * t + params[2]))
        # return 2 * np.pi * (params[1] * t + params[2])
        return (params[1] * (t + 2 * params[2] * np.pi))

    def evaluate(self, params, t):
        result = np.nan
        if len(params.shape) == 1:
            params = list([params])
        for i in range(t.shape[0]):
            cur_temp = self.each_evaluate(params[i], t[i])
            if np.all(np.isnan(result)):
                result = cur_temp
            else:    
                result += cur_temp
        return params[0][0] * np.sin(result)


    def name(self, with_params=False):
        # return self.name_ + str(self.params)
        a = self.params[0][0]
        params_str = ''
        for iter_param in self.params:
            ai, w, fi = iter_param
            params_str += '{}t + {}pi + '.format(round(w, 2), round(fi, 2))

        # a, w, fi = self.params.T # !!!!!
        return '{}Sin({})'.format(round(a, 2), params_str[:-2])
    
    def set_descriptor(self, key: int, descriptor_name: str, descriptor_value):
        descriptor_value = (descriptor_value[0] * 2 * np.pi, descriptor_value[1] * 2 * np.pi)
        return super().set_descriptor(key, descriptor_name, descriptor_value)

    
class Imp(Token):
    def __init__(self, number_params=7, params_description=None, params=None, name='Imp', optimize_id=1):
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
                         params=params, name_=name, optimize_id=optimize_id)
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
                m[cond2] = (np.abs(t1[cond2] - T) / T3) ** p2 #np.abs(p2)
        except:
            m = None
            if cond1:
                m = np.abs((t1 - T1) / T2) ** np.abs(p1)
            elif cond2:
                m = np.abs((t1 - T) / T3) ** np.abs(p2)
        m[np.abs(m) < 0.02] = 0
        return m

    def evaluate(self, params, t):
        # print("checking parameters of Imp", params)
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
        param_str = ''
        for iter_param in self.params:
            w, fi = iter_param[[1, -1]]
            param_str += '{}t + {}pi + '.format(round(w, 2), round(fi, 2))

        return '{}Imp({})'.format(round(a, 2), param_str[:-2])

class Term(Token):
    """
    Class for cashe term of equation
    """

    def __init__(self, type_: str = "Term", name: str = None, expression_token: Token = None, data: np.ndarray = None, mandatory: bool = False) -> None:
        super().__init__(type_=type_, name_=name)
        if expression_token is None:
            expression_token = Constant(params=np.array([1]))
        self._expression_token = expression_token
        assert not data is None, "There must be a field of data"
        self._data = data
        self.mandatory = mandatory

    def __eq__(self, other):
        trm = (self.name_ == other.name_)
        tkn = (self._expression_token == other._expression_token)

        return (tkn and trm)

    def evaluate(self, params, grid: np.ndarray):
        return self._expression_token.value(grid) * self._data

    @property
    def expression_token(self):
        return self._expression_token
    
    @expression_token.setter
    def expression_token(self, new_token):
        assert isinstance(new_token, Token), "New value for expression_token in Term must be type Token"
        self._expression_token = new_token

    def name(self, with_params):
       str_result = '{} {}'
       deq  = self._expression_token.name()
       return str_result.format(deq, self.name_) 
    
    @property
    def data(self):
        return self._data


    
