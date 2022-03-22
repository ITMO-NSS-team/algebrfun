"""
Contains tools for parallelization work.
"""
import functools
from copy import copy
from multiprocessing import Pool, current_process, active_children
import buildingBlocks.Globals.GlobalEntities as Ge
import numpy as np


global pool
pool = None


def create_pool() -> None:
    """
    Set property workers for object of 'WorkersKeeper' class.

    Parameters
    ----------
    workers: int
        Number of workers.

    """
    global pool
    if pool is not None:
        return
    workers = Ge.get_n_jobs()
    if current_process().name == 'MainProcess':
        if workers == 0:
            pool = None
        else:
            pool = Pool(workers)
            print('NEW POOL', flush=True)
            print(pool, flush=True)
            print(current_process(), active_children(), flush=True)


# class MapWrapper:
#
#     def __call__(self, func):
#         if current_process().name != 'MainProcess':
#             @functools.wraps(func)
#             def n_func(self, *args, **kwargs): return []
#             return n_func
#
#         @functools.wraps(func)
#         def wrapper(self, *args, **kwargs):
#             global pool
#             if workersKeeper.workers == 0:
#                 return func(self, kwargs['population'])
#             ret = list(pool.map(functools.partial(func, self),
#                                 np.array_split(kwargs['population'], len(active_children())),
#                                 len(active_children())))
#             ret = [item for sublist in ret for item in sublist]
#             return ret
#         return wrapper

# class MapWrapper:
#
#     def __call__(self, func, *args, **kwargs):
#         if current_process().name != 'MainProcess':
#             return []
#
#         global pool
#         if workersKeeper.workers == 0:
#             return func(self, kwargs['population'])
#         ret = list(pool.map(functools.partial(func, self),
#                             np.array_split(kwargs['population'], len(active_children())),
#                             len(active_children())))
#         ret = [item for sublist in ret for item in sublist]
#         return ret

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
            for i in range(wanted_parts)]


def split_population(population):
    n_jobs = len(active_children())
    split_structures = split_list(population.structure, n_jobs)

    split_populations = []
    for structure in split_structures:
        new_population = type(population)()
        new_population.iterations = population.iterations
        new_population.structure = structure
        split_populations.append(new_population)
    return split_populations


def map_wrapper(func, self, *args, **kwargs):
    """
    Decorator for parallelizing method 'apply' of object of class 'GeneticOperatorPopulation'.

    Parameters
    ----------
    func: method
        Method to wrap.
    self:
        Object with given method.
    kwargs:
        population: list
            List of individuals.
    Returns
    -------
    Wrapped method.
    """
    if current_process().name != 'MainProcess':
        return

    global pool
    population = kwargs['population']
    split_populations = split_population(population)
    workers = Ge.get_n_jobs()
    if workers == 0:
        return func(self, population=population)
    ret_populations = list(pool.map(functools.partial(func, self),
                           split_populations,
                           len(active_children())))
    population.structure = []
    for ret_pop in ret_populations:
        population.add_substructure(ret_pop.structure)
    return
