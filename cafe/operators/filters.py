import numpy as np
from itertools import groupby

from .base import GeneticOperatorIndivid
from .base import GeneticOperatorPopulation
from .base import apply_decorator

def del_duplicate_elements(structure):
    new_chromo = []

    for elem in structure:
        if elem not in new_chromo:
            new_chromo.append(elem)
    
    return new_chromo

class FilterIndivid(GeneticOperatorIndivid):
    """
    """
    def __init__(self, params: dict = None):
        super().__init__(params)
            

    @apply_decorator
    def apply(self, individ, *args, **kwargs):
        # expressions = groupby(individ.structure, key=lambda term: term.name_)
        names = np.unique(list(map(lambda elem: elem.name_, individ.structure)))
        new_chromo = []

        for key in names:
            tokens = list(filter(lambda elem: elem.name_ == key, individ.structure))
            list_of_tokens = del_duplicate_elements(tokens)
            new_chromo.extend(list_of_tokens[:individ.max_tokens]) # обрезаем по кол-ву элементов в алг выражении
            
        individ.structure = new_chromo

class FilterPopulation(GeneticOperatorPopulation):
    """
    """

    def __init__(self, params: dict = None):
        super().__init__(params)

    def apply(self, population, *args, **kwargs) -> None:
        new_structure = del_duplicate_elements(population.structure)
        if self.params['population_size'] < len(new_structure):
            elite = list(filter(lambda ind: ind.elitism, new_structure))

            for individ in elite:
                new_structure.remove(individ)
            
            population_fitnesses = list(map(lambda ind: 1/(ind.fitness+0.01), new_structure))
            fits_sum = np.sum(population_fitnesses)
            probabilities = list(map(lambda x: x / fits_sum, population_fitnesses))
            new_structure = list(np.random.choice(new_structure, size=self.params['population_size'],
                                                         p=probabilities, replace=False))
            new_structure.extend(elite)
        
        population.structure = new_structure