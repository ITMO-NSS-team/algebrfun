"""Костыль, кладет параметры заданные пользователем в генетические операторы"""
# todo придумать нормальный оператор билдер для удобства пользователю с приятным интерфейсом

from buildingBlocks.baseline.OperatorsKeepers import OperatorsKeeper
from buildingBlocks.default.EvolutionEntities import Equation
from buildingBlocks.default.EvolutionEntities import PopulationOfEquations
from buildingBlocks.default.geneticOperators import (Crossovers, FitnessEvaluators,
                                                     Initializers, Mutations, Regularizations,
                                                     Selectors)
from buildingBlocks.default.geneticOperators.ComplexOptimizers import ImpComplexDiscreteTokenParamsOptimizer, \
    AllImpComplexOptimizerIndivid, ImpComplexOptimizerIndivid, ImpSimpleOptimizerIndivid, \
    ImpComplexTokenParamsOptimizer, ImpComplexOptimizerIndivid2
from buildingBlocks.default.geneticOperators.FitnessEvaluators import TokenFitnessIndivid
from buildingBlocks.default.geneticOperators.Optimizers import PeriodicTokensOptimizerIndivid, \
    PeriodicCAFTokensOptimizerPopulation, PeriodicInProductTokensOptimizerIndivid, PeriodicExtraTokensOptimizerIndivid, \
    TrendTokensOptimizerIndivid, TrendDiscreteTokensOptimizerIndivid, DifferentialTokensOptimizerPopulation, ParamsOfEquationOptimizerIndivid

import buildingBlocks.Globals.GlobalEntities as Bg

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.optimize import minimize

# from moea_dd.forMoeadd.loadTS import ts

# Определим число процессов
from buildingBlocks.default.geneticOperators.Regularizations import RestrictTokensIndivid, LassoIndivid
from buildingBlocks.default.geneticOperators.Unifiers import UnifierParallelizedPopulation, UnifierIndivid


def full_update(d: dict):
    pass


default = {
    'mutation': {
        'simple': dict(intensive=1, increase_prob=1),
        'complex': dict(prob=0.2, threshold=0.1)
    },
    'crossover': {
        'simple': dict(intensive=1, increase_prob=0.3)
    },
    'tokens': None,
    'regularization': {
        'max_tokens': 5
    }
}


def set_operators(grid, individ, kwargs):
    mutation = kwargs['mutation']
    crossover = kwargs['crossover']
    population_size = kwargs['population']['size']
    tokens = kwargs['tokens']
    lasso = kwargs['lasso']

    for token in tokens:
        token.__select_parametrs__(grid, Bg.constants['target'], population_size, gen=False)


    operatorsMap = OperatorsKeeper()


    operatorsMap.VarFitnessIndivid = FitnessEvaluators.VarFitnessIndivid(
        params=dict(grid=grid))

    operatorsMap.FitnessPopulation = FitnessEvaluators.FitnessPopulation()

    operatorsMap.InitIndivid = Initializers.InitIndivid(params=dict(tokens=tokens, grid=grid))

    operatorsMap.InitPopulation = Initializers.InitPopulation(
        params=dict(population_size=population_size,
                    individ=individ))
    
    operatorsMap.InitSubPopulation = Initializers.InitSubPopulations(params=dict(population_size=population_size,
                    individ=individ))

    operatorsMap.MutationIndivid = Mutations.MutationIndivid(
        params=dict(mut_intensive=mutation['simple']['intensive'],
                    increase_prob=mutation['simple']['increase_prob'],
                    tokens=tokens))

    operatorsMap.ImpComplexMutationIndivid = Mutations.ImpComplexMutationIndivid(
        params=dict(mut_prob=mutation['complex']['prob'],
                    complex_token=mutation['complex']['complex_token'],
                    grid=grid,
                    threshold=mutation['complex']['threshold']))

    operatorsMap.MutationPopulation = Mutations.MutationPopulation(
        params=dict(mutation_size=int(0.3*population_size)+1))

    operatorsMap.CrossoverIndivid = Crossovers.CrossoverIndivid(
        params=dict(cross_intensive=crossover['simple']['intensive'],
                    increase_prob=crossover['simple']['increase_prob']))

    operatorsMap.CrossoverPopulation = Crossovers.CrossoverPopulation(
        params=dict(crossover_size=int(0.4*population_size)+1))

    operatorsMap.RouletteWheelSelection = Selectors.RouletteWheelSelection(
        params=dict(tournament_size=population_size,
                    winners_size=int(0.5*population_size)+1))

    operatorsMap.DelDuplicateTokensIndivid = Regularizations.DelDuplicateTokensIndivid()

    operatorsMap.CheckMandatoryTokensIndivid = Regularizations.CheckMandatoryTokensIndivid(
            params=dict(tokens=tokens,
                        add_to_complex_prob=0.5))

    operatorsMap.RegularisationPopulation = Regularizations.RegularisationPopulation(
        params=dict(parallelise=False))

    operatorsMap.Elitism = Selectors.Elitism(
        params=dict(elitism=1))

    operatorsMap.RestrictPopulation = Selectors.RestrictPopulation(
        params=dict(population_size=population_size))

    operatorsMap.LassoIndivid1Target = Regularizations.LassoIndivid1Target(
        params=dict(grid=grid,
                    regularisation_coef=0.01))

    operatorsMap.LRIndivid1Target = Regularizations.LRIndivid1Target(
        params=dict(grid=grid))

    # operatorsMap.lassoIndivid = Regularizations.DEOptIndivid(params=dict(grid=grid))

    operatorsMap.LassoPopulation = Regularizations.LassoPopulation(
        params=dict(parallelise=False))

    # operatorsMap.ProductTokenMutationIndivid = Mutations.ProductTokenMutationIndivid(
    #     params=dict(tokens=p_tokens,
    #                 product_token=product_token,
    #                 mut_prob=0.,
    #                 max_multi_len=maxlen_subtokens))

    operatorsMap.PeriodicTokensOptimizerIndivid = PeriodicTokensOptimizerIndivid(
          params=dict(grid=grid,
                    optimize_id=1,
                    optimizer='DE',
                    popsize=10))

    operatorsMap.PeriodicExtraTokensOptimizerIndivid = PeriodicExtraTokensOptimizerIndivid(
        params=dict(grid=grid,
                    optimize_id=1))

    operatorsMap.TrendTokensOptimizerIndivid = TrendTokensOptimizerIndivid(
        params=dict(grid=grid,
                    optimize_id=2))

    operatorsMap.TrendDiscreteTokensOptimizerIndivid = TrendDiscreteTokensOptimizerIndivid(
        params=dict(grid=grid,
                    optimize_id=2,
                    optimizer='DE',
                    popsize=10))

    operatorsMap.ParamsOfEquationOptimizerIndivid = ParamsOfEquationOptimizerIndivid(
        params=dict(grid=grid, popsize=10)
    )

    # operatorsMap.PeriodicInProductTokensOptimizerIndivid = PeriodicInProductTokensOptimizerIndivid(
    #     params=dict(grid=grid,
    #                 optimize_id=1,
    #                 optimizer='DE',
    #                 complex_tokens_types=[type(product_token)]))

    operatorsMap.ImpComplexDiscreteTokenParamsOptimizer = ImpComplexDiscreteTokenParamsOptimizer(
        params=dict(grid=grid,
                    optimize_id=3,
                    optimizer='DE',
                    popsize=None))

    operatorsMap.PeriodicCAFTokensOptimizerPopulation = PeriodicCAFTokensOptimizerPopulation(
        params=dict(parallelise=False))

    operatorsMap.UnifierParallelizedPopulation = UnifierParallelizedPopulation(
        params=dict(parallelise=False))

    operatorsMap.TokenFitnessIndivid = TokenFitnessIndivid()

    operatorsMap.RestrictTokensIndivid = RestrictTokensIndivid()

    operatorsMap.LassoIndivid = LassoIndivid(
        params=dict(grid=grid,
                    regularisation_coef=lasso['regularisation_coef'])
    )

    operatorsMap.AllImpComplexOptimizerIndivid = AllImpComplexOptimizerIndivid(
        params=dict(grid=grid,
                    optimize_id=3)
    )

    operatorsMap.ImpComplexOptimizerIndivid = ImpComplexOptimizerIndivid(
        params=dict(grid=grid)
    )

    operatorsMap.ImpSimpleOptimizerIndivid = ImpSimpleOptimizerIndivid(
        params=dict(grid=grid,
                    constraints=None)
    )

    operatorsMap.ImpComplexOptimizerIndivid2 = ImpComplexOptimizerIndivid2(
        params=dict(grid=grid,
                    optimize_id=3)
    )

    operatorsMap.UnifierIndivid = UnifierIndivid()

    operatorsMap.DifferentialTokensOptimizerPopulation = DifferentialTokensOptimizerPopulation()

# Загружаем операторы в глобальную переменную чтобы ими могли пользоваться все индивиды
    Bg.set_operators(operatorsMap)
