# from buildingBlocks.baseline.GeneticOperators import GeneticOperatorPopulation
from buildingBlocks.baseline.BasicEvolutionaryEntities import GeneticOperatorPopulation
import numpy as np


class Elitism(GeneticOperatorPopulation):
    """
    Marks the given number of the best individs. Change property self.elitism.
    """
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('elitism')

    def apply(self, population, *args, **kwargs):
        for individ in population.structure:
            individ.elitism = False
        elite_idxs = np.argsort(list(map(lambda ind: ind.fitness,
                                         population.structure)))[:self.params['elitism']]
        for idx in elite_idxs:
            population.structure[idx].elitism = True
        return population


class RouletteWheelSelection(GeneticOperatorPopulation):
    """
    Marks the given number of the selected individs. Change property self.selected.
    """
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('tournament_size', 'winners_size')

    def apply(self, population, *args, **kwargs):
        # pops = list(filter(lambda ind: len(ind.structure) > 1, population.structure)) # может быть [] (так как удаляются
        # все не мандатори токены, например периодические если не было найдено нормальной частоты)
        # assert len(pops) != 0, 'All individs in population consist of one token (probably because ' \
        #                        'all non-mandatory tokens were deleted during optimization or regularization)'

        pops = population.structure
        assert len(pops) > 0, 'Empty population'
        for individ in pops:
            individ.selected = False

        tournament_size = self.params['tournament_size']
        winners_size = self.params['winners_size']
        if tournament_size is None:
            tournament_size = len(pops)
            selected_individs = pops
        # else:
            # assert tournament_size <= len(pops), "Tournament size must be less than population size"
            # tournament_size = min(tournament_size, len(pops))
        if tournament_size > len(pops):
            tournament_size = len(pops)
            winners_size = round(tournament_size / 100 * 60)
        assert tournament_size <= len(pops), "Tournament size must be less than population size"
        selected_individs = list(np.random.choice(pops, replace=False, size=tournament_size))
        population_fitnesses = list(map(lambda ind: 1/(ind.fitness + 0.01), selected_individs))
        fits_sum = np.sum(population_fitnesses)
        probabilities = list(map(lambda x: x/fits_sum, population_fitnesses))
        if winners_size is None:
            winners = selected_individs
        else:
            assert tournament_size >= winners_size, "Winners size must be less than tornament size"
            try:
                winners = np.random.choice(selected_individs, size=winners_size, p=probabilities, replace=False)
            except:
                winners = np.random.choice(selected_individs, size=winners_size, replace=False)
        # return list(winners)
        for individ in winners:
            individ.selected = True
        return population


class RestrictPopulation(GeneticOperatorPopulation):
    """
    Restrict the size of the population to the given population_size according to the fitness-dependent probability.
    """
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('population_size')

    def apply(self, population, *args, **kwargs):
        if self.params['population_size'] < len(population.structure):
            elite = list(filter(lambda ind: ind.elitism, population.structure))
            # Исключаем элиту чтобы позже ее добавить
            for individ in elite:
                population.structure.remove(individ)
            population_fitnesses = list(map(lambda ind: 1/(ind.fitness+0.01), population.structure))
            fits_sum = np.sum(population_fitnesses)
            probabilities = list(map(lambda x: x / fits_sum, population_fitnesses))
            population.structure = list(np.random.choice(population.structure, size=self.params['population_size'],
                                                         p=probabilities, replace=False))
            population.structure.extend(elite)
        return population


class FiltersPopulationOfDEquation(GeneticOperatorPopulation):
    def __init__(self, params):
        super().__init__(params=params)
    
    def apply(self, population, *args, **kwargs):
        new_population_structure = []
        for individ in population.structure:
            mandatory_tokens = list(filter(lambda token: token.mandatory != 0, individ.structure))
            if len(mandatory_tokens) != 0:
                current_structure = individ.structure
                individ.structure = []
                individ.set_structure(current_structure)
                new_population_structure.append(individ)
        
        population.structure = new_population_structure

        return population

