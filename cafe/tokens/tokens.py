import numpy as np

from base import Token

class Constant(Token):
    """
    Basic function for decript 
    """
    def __init__(self, number_params: int = 1, params_decription: dict = ..., params: np.ndarray = None, val: np.ndarray = None, name_: str = None, optimize_id: int = 2) -> None:
        if params_decription is None:
            params_decription = {
                0: dict(name='Amplitude', bounds=(-100, 100)),
            }
        super().__init__(number_params, params_decription, params, val, name_, optimize_id)

        self.type = 'Constant'

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
                 params=None, name_='Power', optimize_id=2):
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
                 params=None, name_='Sin', optimize_id=1):
        if params_description is None:
            params_description = {
                0: dict(name='Amplitude', bounds=(0., 10.)),
                1: dict(name='Frequency', bounds=(0., float('inf'))),
                2: dict(name='Phase', bounds=(0., 3.))
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
        # return np.sin(2 * np.pi * (params[1] * t + params[2]))
        # return 2 * np.pi * (params[1] * t + params[2])
        return (params[1] * t + params[2] * np.pi)

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

    
