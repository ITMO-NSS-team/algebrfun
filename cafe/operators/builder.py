
from .base import OperatorMap

from .initializers import InitIndivid
from .initializers import InitPopulation


def create_operator_map(grid, individ, kwargs):
    mutation = kwargs['mutation']
    crossover = kwargs['crossover']
    population_size = kwargs['population']['size']
    tokens = kwargs['tokens']
    lasso = kwargs['lasso']
    terms = kwargs['terms']
    shape = kwargs['shape']

    operatorsMap = OperatorMap()

    operatorsMap.InitIndivid =  InitIndivid(params=dict(tokens=tokens, terms=terms, grid=grid))

    operatorsMap.InitPopulation = InitPopulation(
        params=dict(population_size=population_size,
                    individ=individ))