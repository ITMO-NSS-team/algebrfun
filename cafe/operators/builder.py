
from .base import OperatorMap

from .initializers import InitIndivid
from .initializers import InitPopulation

from .optimizers import TokenParametersOptimizerPopulation
from .optimizers import TokenParametersOptimizerIndivid

from .fitness import VarFitnessIndivid
from .fitness import TokenFitnessIndivid
from .fitness import FitnessPopulation

from .filters import FilterIndivid
from .filters import FilterPopulation

from .regularizations import LRIndivid
from .regularizations import DecimationPopulation

from .selectors import Elitism
from .selectors import RouletteWheelSelection

from .crossover import CrossoverIndivid
from .crossover import CrossoverPopulation

from .mutation import MutationIndivid
from .mutation import MutationPopulation


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
        params=dict(grid=grid, shape=shape, target=kwargs['target']._data)
    )

    operatorsMap.VarFitnessIndivid = VarFitnessIndivid(
        params=dict(grid=grid)
    )

    operatorsMap.TokenFitnessIndivid = TokenFitnessIndivid()

    operatorsMap.FilterIndivid = FilterIndivid()

    operatorsMap.LRIndivid = LRIndivid(
        params=dict(grid=grid)
    )

    operatorsMap.FitnessPopulation = FitnessPopulation()

    operatorsMap.FilterPopulation = FilterPopulation(
        params=dict(population_size=population_size)
    )

    operatorsMap.Elitism = Elitism(
        params=dict(elitism=1)
    )

    operatorsMap.RouletteWheelSelection = RouletteWheelSelection(
        params=dict(tournament_size=population_size, winners_size=int(0.5*population_size)+1)
    )

    operatorsMap.CrossoverPopulation = CrossoverPopulation(
        params=dict(crossover_size=int(0.4*population_size)+1)
    )

    operatorsMap.CrossoverIndivid = CrossoverIndivid(
        params=dict(cross_intensive=crossover['simple']['intensive'],
                    increase_prob=crossover['simple']['increase_prob'])
    )

    operatorsMap.MutationPopulation = MutationPopulation(
        params=dict(mutation_size=int(0.3*population_size)+1)
    )

    operatorsMap.MutationIndivid = MutationIndivid(
        params=dict(mut_intensive=mutation['simple']['intensive'],
                    increase_prob=mutation['simple']['increase_prob'],
                    tokens=tokens)
    )

    operatorsMap.DecimationPopulation = DecimationPopulation(
        params=dict(grid=grid)
    )