from .base import GeneticOperatorPopulation
import numpy as np

class Elitism(GeneticOperatorPopulation):
    """
    """

    def __init__(self, params: dict = None):
        super().__init__(params)

    def apply(self, population, *args, **kwargs):
        for individ in population.structure:
            individ.elitism = False
        elite_idxs = np.argsort(list(map(lambda ind: ind.fitness,
                                         population.structure)))[:self.params['elitism']]
        
        for idx in elite_idxs:
            population.structure[idx].elitism = True
            population.anal.append(population.structure[idx].fitness)

class RouletteWheelSelection(GeneticOperatorPopulation):
    """
    """

    def __init__(self, params: dict = None):
        super().__init__(params)
    
    def apply(self, population, *args, **kwargs) -> None:
        pops = population.structure
        assert len(pops) > 0, 'Empty population'
        for individ in pops:
            individ.selected = False
        
        tournament_size = self.params['tournament_size']
        winners_size = self.params['winners_size']

        if tournament_size is None or tournament_size > len(pops):
            tournament_size = len(pops)
            winners_size = round(tournament_size / 100 * 50)
    

        selected_individs = list(np.random.choice(pops, replace=False, size=tournament_size))
        population_fitnesses = list(map(lambda ind: 1/(ind.fitness + 0.01), selected_individs))
        fits_sum = np.sum(population_fitnesses)
        probabilities = list(map(lambda x: x/fits_sum, population_fitnesses))

        if winners_size is None or winners_size == 0:
            winners = selected_individs
        else:
            try:
                winners = np.random.choice(selected_individs, size=winners_size, p=probabilities, replace=False)
            except:
                winners = np.random.choice(selected_individs, size=winners_size, replace=False)
        
        for individ in winners:
            individ.selected = True
