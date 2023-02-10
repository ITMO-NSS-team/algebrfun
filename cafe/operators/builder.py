
from .base import OperatorMap

from .initializers import InitIndivid
from .initializers import InitPopulation

from .optimizers import TokenParametersOptimizerPopulation
from .optimizers import TokenParametersOptimizerIndivid

from .fitness import VarFitnessIndivid
from .fitness import TokenFitnessIndivid

from .filters import FilterIndivid

from .regularizations import LRIndivid


def create_operator_map(grid, individ, kwargs):
    mutation = kwargs['mutation']
    crossover = kwargs['crossover']
    population_size = kwargs['population']['size']
    tokens = kwargs['tokens']
    lasso = kwargs['lasso']
    terms = kwargs['terms']
    shape = kwargs['shape']

    for token in tokens:
        token._find_initial_approximation_(grid, kwargs['target']._data, population_size, gen=False)

    operatorsMap = OperatorMap()

    operatorsMap.InitIndivid =  InitIndivid(params=dict(tokens=tokens, terms=terms, grid=grid, shape=shape))

    operatorsMap.InitPopulation = InitPopulation(
        params=dict(population_size=population_size,
                    individ=individ))

    operatorsMap.TokenParametersOptimizerPopulation = TokenParametersOptimizerPopulation()

    operatorsMap.TokenParametersOptimizerIndivid = TokenParametersOptimizerIndivid(
        params=dict(grid=grid, shape=shape)
    )

    operatorsMap.VarFitnessIndivid = VarFitnessIndivid(
        params=dict(grid=grid)
    )

    operatorsMap.TokenFitnessIndivid = TokenFitnessIndivid()

    operatorsMap.FilterIndivid = FilterIndivid()

    operatorsMap.LRIndivid = LRIndivid(
        params=dict(grid=grid)
    )