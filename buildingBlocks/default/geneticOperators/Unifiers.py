"""Операторы, которые объединяют в себе воздействие нескольких операторов"""

from buildingBlocks.baseline.BasicEvolutionaryEntities import GeneticOperatorIndivid, GeneticOperatorPopulation


class UnifierIndivid(GeneticOperatorIndivid):
    def __init__(self, params=None):
        super().__init__(params=params)

    def apply(self, individ, *args, **kwargs):

        individ.apply_operator('CheckMandatoryTokensIndivid')

        individ.apply_operator('TrendDiscreteTokensOptimizerIndivid')
        individ.apply_operator('ImpComplexDiscreteTokenParamsOptimizer')
        individ.apply_operator('PeriodicTokensOptimizerIndivid')

        individ.apply_operator('TokenFitnessIndivid')
        individ.apply_operator('DelDuplicateTokensIndivid')
        individ.apply_operator('RestrictTokensIndivid')
        individ.apply_operator('LRIndivid1Target')


class UnifierParallelizedPopulation(GeneticOperatorPopulation):
    def __init__(self, params=None):
        super().__init__(params=params)

    def apply(self, population, *args, **kwargs):
        population.apply_operator('PeriodicTokensOptimizerPopulation')
        population.apply_operator('RegularisationPopulation')
        population.apply_operator('FitnessPopulation')
        return population
