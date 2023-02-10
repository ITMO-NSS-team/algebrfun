import numpy as np
from itertools import groupby

from .base import GeneticOperatorIndivid
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
        expressions = groupby(individ.structure, key=lambda term: term.name)
        new_chromo = []

        for one_term in expressions:
            list_of_tokens = del_duplicate_elements(one_term)
            new_chromo.extend(list_of_tokens[:individ.max_tokens]) # обрезаем по кол-ву токено в алг выражении
            
        individ.structure = new_chromo
