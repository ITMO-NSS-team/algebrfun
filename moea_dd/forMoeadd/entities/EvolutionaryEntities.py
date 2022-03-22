"""Тут наследники базовых сущностей однокритериального алгоритма, дополнительно модифицированные для
многокритериального алгоритма"""

from copy import copy
import numpy as np

import buildingBlocks.default.EvolutionEntities as Ee
import buildingBlocks.Globals.GlobalEntities as Bg

import moea_dd.src.moeadd as moeadd
import moea_dd.src.moeadd_supplementary as moeadd_sup


class MultiEquation(Ee.Equation):

    def __init__(self, structure: list = None,
                 fixator=None,
                 fitness: float = None,
                 used_value: str = 'plus', forms=None,
                 lasso_coef: float = None):
        super().__init__(structure=structure, fixator=fixator,
                         fitness=fitness, used_value=used_value, forms=forms)
        self.lasso_coef = lasso_coef

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if len(self.structure) == len(other.structure):
                for token in self.structure:
                    if self.structure.count(token) != other.structure.count(token):
                        return False
                return True
        return False


class MoeaddIndividTS(moeadd.moeadd_solution):

    def __init__(self, x, obj_funs: list):
        super().__init__(x, obj_funs)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.vals == other.vals
        return False

    def __hash__(self):
        return id(self)

    def copy(self):
        ind_copy = copy(self)
        ind_copy.vals = ind_copy.vals.copy()
        return ind_copy


global count
count = 0


class PopulationConstructor:
    def __init__(self, pattern):
        self.pattern = pattern

    def create(self, *args):
        global count
        created_individ = self.pattern.copy()

        created_individ.vals.max_tokens = np.random.randint(2, 15)

        created_individ.vals.apply_operator('InitIndivid')

        created_individ.vals.apply_operator('UnifierIndivid')

        global count
        count += 1
        # print('\n\n--------------->>{}<<--------------\n\n'.format(count))

        return created_individ


class EvolutionaryOperator:

    def __init__(self):
        pass

    def mutation(self, individ):
        mutant = individ.copy()
        assert mutant is not individ, 'need to deepcopy'

        if np.random.uniform() <= 0.5:
            mutant.vals.max_tokens += np.random.randint(-1, 2)
            mutant.vals.max_tokens = max(2, mutant.vals.max_tokens)

        mutant.vals.apply_operator('MutationIndivid')
        mutant.vals.apply_operator('ImpComplexMutationIndivid')

        mutant.vals.apply_operator('UnifierIndivid')
        return mutant

    def crossover(self, parents_pool):
        offspring_pool = []
        for idx in np.arange(np.int(np.floor(len(parents_pool) / 2.))):
            offsprings = [parents_pool[2 * idx].copy(), parents_pool[2 * idx + 1].copy()]

            for idx, ofsp in enumerate(offsprings):
                assert ofsp is not parents_pool[idx], 'need to deepcopy'

            offsprings[0].vals.apply_operator('CrossoverIndivid', other_individ=offsprings[1].vals)

            # prob = np.random.uniform()
            # if prob <= 1/2.:
            #     offsprings[0].vals.max_tokens, offsprings[1].vals.max_tokens = \
            #         offsprings[1].vals.max_tokens, offsprings[0].vals.max_tokens
            # elif 1/3.<prob<=2/3.:
            #     new_coef = (offsprings[0].lasso_coef + offsprings[1].lasso_coef)/2
            #     offsprings[0].lasso_coef, offsprings[1].lasso_coef = new_coef, new_coef
            # else:
            #     pass

            for offspring in offsprings:
                offspring.vals.apply_operator('UnifierIndivid')

                offspring.precomputed_value = False
                offspring.precomputed_domain = False

            offspring_pool.extend(offsprings)
        return offspring_pool