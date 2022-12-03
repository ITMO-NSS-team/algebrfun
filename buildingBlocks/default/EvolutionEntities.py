"""
Contains default inheritors/implementations of baseline classes for Individual.
"""
from ast import Pass
from copy import copy, deepcopy
from ctypes import Structure
from functools import reduce
from buildingBlocks.baseline.BasicEvolutionaryEntities import Individ, Population
import numpy as np
from buildingBlocks.Globals.GlobalEntities import set_constants, get_full_constant
from buildingBlocks.baseline.BasicEvolutionaryEntities import DifferentialToken


class Equation(Individ):
    """
    An expression that approximates the input data and is used as a basis for building synthetics.
    """
    def __init__(self, structure: list = None,
                 fixator=None,
                 fitness: float = None,
                 used_value: str = 'plus', forms=None,
                 max_tokens: int = None):
        super().__init__(structure=structure, fixator=fixator, fitness=fitness)
        if forms is None:
            forms = []
        self.elitism = False
        self.selected = False

        self.intercept = 0.

        # self.kind = kind
        self.used_value = used_value
        self.forms = forms

        self.max_tokens = max_tokens
        self.type_ = "Equation"
        self.owner_id = None

    def __getstate__(self):
        # for key in self.__dict__.keys():
        #     if key in ('val',):
        #         self.__dict__[key] = None
        #     if key in ('forms',):
        #         self.__dict__[key] = []
        return self.__dict__

    def __setstate__(self, state: dict):
        self.__dict__ = state

    def copy(self):
        new_copy = deepcopy(self)

        try:
            new_copy.forms = deepcopy(new_copy.forms)
        except:
            pass

        return new_copy

    def clean_copy(self):
        new_copy = type(self)()
        return new_copy

    def formula(self, with_params=False):
        if self.used_value == "plus":
            joinWith = '+'
        else:
            joinWith = '*'
        return joinWith.join(list(map(lambda x: x.name(with_params), self.structure)))

    def value(self, grid):
        fixed_optimized_tokens_in_structure = list(filter(lambda token: token.fixator['self'], self.structure))
        # print("checking value in equation", fixed_optimized_tokens_in_structure)
        if len(fixed_optimized_tokens_in_structure) != 0:
            # if self.used_value == 'plus':
            # вычисляем валуе только от оптимизированных токенов
            # print('Mean values of f-ing tokens', [np.mean(token.value(grid)) for token in fixed_optimized_tokens_in_structure])
            val = reduce(lambda val, x: val + x, list(map(lambda x: x.value(grid),
                                                          fixed_optimized_tokens_in_structure)))
            # val -= self.intercept
            # val = val.reshape(grid.shape) 
            # print("returned value",  val)
            return val
        #     elif self.used_value == 'product':
        #         return reduce(lambda val, x: val * x, list(map(lambda x: x.value(t), self.chromo)))
        return np.zeros((grid.shape[-1],))

    def get_norm_of_amplitudes(self):
        fixed_optimized_tokens_in_structure = list(filter(lambda token: token.fixator['self'], self.structure))

        if len(fixed_optimized_tokens_in_structure) != 0:
            norm = reduce(lambda norm, ampl: norm + ampl, list(map(lambda token: np.abs(token.param(name='Amplitude')[0]), fixed_optimized_tokens_in_structure)))

            return norm
        
        return 0

    def get_structure(self, *types):
        if not types:
            return copy(self.structure)
        return list(filter(lambda x: x.type in types, self.structure))


class PopulationOfEquations(Population):
    """
    Class with population of Equations.
    """
    def __init__(self, structure: list = None,
                 iterations: int = 0):
        super().__init__(structure=structure)

        self.iterations = iterations
        self.type_ = "PopulationOfEquations"
        self._owner_id = None
        # self.loggers = [Logger(), Logger()]

    @property
    def owner_id(self):
        return self._owner_id

    @owner_id.setter
    def owner_id(self, value: int):
        self._owner_id = value
        for ind in self.structure:
            ind.owner_id = value

    def _evolutionary_step(self):
        # self.apply_operator('RegularisationPopulation')
        # self.apply_operator('PeriodicTokensOptimizerPopulation')
        # self.apply_operator('LassoPopulation')
        # self.apply_operator('FitnessPopulation')
        # self.text_indiv_param()
        self.apply_operator('UnifierParallelizedPopulation')
        # self.text_indiv_param()

        self.apply_operator('RestrictPopulation')
        self.apply_operator('Elitism')
        self.apply_operator('RouletteWheelSelection')
        self.apply_operator('CrossoverPopulation')
        self.apply_operator('MutationPopulation')

    def evolutionary(self, *args):
        # self.apply_operator('InitPopulation')
        constants = get_full_constant()
        exist_ids = [tkn.params[1].term_id for tkn in constants['best_individ'].structure]
        if not self.owner_id in exist_ids:
            return
        for n in range(self.iterations):
            # print('{}/{}\n'.format(n, self.iterations))
            self._evolutionary_step()
            idxsort = np.argsort(list(map(lambda x: x.fitness, self.structure)))
            inds = [self.structure[i] for i in idxsort]
            # print("structure of population", [cur_eq.fitness for cur_eq in inds])
            # print("no sort", [cur_eq.fitness for cur_eq in self.structure])
        # self.apply_operator('RegularisationPopulation')
        # self.apply_operator('PeriodicTokensOptimizerPopulation')
        # self.apply_operator('LassoPopulation')
        # self.apply_operator('FitnessPopulation')
        self.apply_operator('UnifierParallelizedPopulation')
        self.apply_operator('RestrictPopulation')

        self.apply_operator('Elitism')


class DEquation(Individ):
    '''
        class for displays individ as differncial equation
    '''
    def __init__(self, structure: list = None, fixator: dict = None, used_value: str = 'plus', forms=None, type_: str = "DEquation", fitness: float = None, max_tokens: int = None, der_set: list = None):
        super().__init__(structure, fixator, fitness)
        
        if forms is None:
                forms = []
        self.elitism = False
        self.selected = False

        self.intercept = 0.

        # self.kind = kind
        self.used_value = used_value
        self.forms = forms

        self.max_tokens = max_tokens
        if der_set is None:
            self.der_set = []
        self.type_ = type_
    
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state: dict):
        self.__dict__ = state

    def set_structure(self, structure):
        print("START", [elem.name() for elem in structure])
        self.change_all_fixes(False)
        assert type(structure) == list, "structure must be a list"
        term_ids = np.unique([token.params[1].term_id for token in structure])
        print("terms unique", term_ids)
        correct_form_structure = []
        for term_id in term_ids:
            tokens_of_caf = list(filter(lambda x: x.params[1].term_id == term_id, structure))
            print(term_id, [elem.name() for elem in tokens_of_caf])
            if len(tokens_of_caf) > 1:
                correct_form_structure.append(type(tokens_of_caf[0])(number_params=2, params_description={0: dict(name='Close algebr equation'), 1: dict(name="Term")}, params=np.array([Equation(structure=[x.params[0].structure[0] for x in tokens_of_caf], fixator=tokens_of_caf[0].params[0].fixator), tokens_of_caf[0].params[1]], dtype=object), name_="DifferencialToken"))
            else:
                correct_form_structure.append(tokens_of_caf[0])
        print([elem.name() for elem in correct_form_structure])
        self.structure = correct_form_structure
        print("END")




    def check_duplicate_of_term(self, new_structure):
        uses = [item.params[1].term_id for item in self.structure]
        structure = []

        try:
            for tkn in new_structure:
                if tkn.params[1].term_id in uses:
                    continue
                uses.append(tkn.params[1].term_id)
                structure.append(tkn)
        except:
            if new_structure.params[1].term_id in uses:
                return structure
            else:
                structure.append(new_structure)
        return structure

    def add_substructure(self, substructure, idx: int = -1) -> None:
        substructure = self.check_duplicate_of_term(substructure)
        super().add_substructure(substructure, idx)


    def set_CAF(self, CAF):
        for tkn in self.structure:
            if tkn.params[1].term_id == CAF.owner_id:
                tkn.params = np.array([CAF, tkn.params[1]])
                return

    def copy(self):
        new_copy = deepcopy(self)

        try:
            new_copy.forms = deepcopy(new_copy.forms)
        except:
            pass

        return new_copy

    def clean_copy(self):
        new_copy = type(self)()
        return new_copy

    def formula(self, with_params=False):
        pass
        if self.used_value == "plus":
            joinWith = '+'
        else:
            joinWith = '*'
        
        return joinWith.join(list(map(lambda x: x.name(with_params), self.structure)))
        

    def value(self, grid):
        # fixed_optimized_tokens_in_structure = list(filter(lambda token: token.fixator['self'], self.structure))

        fixed_optimized_tokens_in_structure = self.structure

        if len(fixed_optimized_tokens_in_structure) != 0:
            value = reduce(lambda val, x: val + x, list(map(lambda x: x.value(grid), fixed_optimized_tokens_in_structure)))

            return value
        
        return None
            
    def get_CAF(self):
        CAFs = []
        for token in self.structure:
            CAFs.append(token.params[0])

        return CAFs


    def get_norm_of_amplitudes(self):
        pass

    def get_structure(self, *types):
        # if not types:
        #     return copy(self.structure)
        # return list(filter(lambda x: x.type in types, self.structure))
        structure_without_brackets = []
        for dif_token in self.structure:
            for token in dif_token.params[0].structure:
                new_part_token = type(dif_token)(number_params=2, params_description={0: dict(name='Close algebr equation'), 1: dict(name="Term")}, params=np.array([Equation(structure=[token]), dif_token.params[1]], dtype=object), name_="DifferencialToken")
                new_part_token.mandatory = dif_token.mandatory
                structure_without_brackets.append(new_part_token)
        
        return structure_without_brackets



class PopulationOfDEquations(Population):
    '''
        class for displays population of individs-DE
    '''
    def __init__(self, structure: list = None, iterations: int = 0, type_: str = "PopulationOfDEquation"):
        super().__init__(structure)

        self.iterations = iterations
        self.type_ = type_
        # self.coef_set = PopulationOfEquations(iterations=1)

    def check(self):
        for ind in self.structure:
            if len(list(filter(lambda token: token.mandatory == 1, ind.structure))) == 0:
                return False
        return True

    
    def _evolutionary_step(self, *args):
        self.apply_operator("DifferentialTokensOptimizerPopulation", args[0])
        if not self.check():
                print("DifferentialTokensOptimizerPopulation")
                exit(1)
        for individ in self.structure:
            individ.apply_operator("LassoIndivid")
            if not self.check():
                print("LassoIndivid")
                print(individ.formula())
                exit(1)
            individ.apply_operator("VarFitnessIndivid")
            if not self.check():
                print("VarFitnessIndivid")
                exit(1)
        self.apply_operator("RouletteWheelSelection")
        if not self.check():
                print("RouletteWheelSelection")
                exit(1)
        self.apply_operator("CrossoverPopulation")
        if not self.check():
                print("CrossoverPopulation")
                exit(1)
        self.apply_operator("FitnessPopulation")
        if not self.check():
                print("FitnessPopulation")
                exit(1)
        self.apply_operator("Elitism")
        if not self.check():
                print("Elitism")
                exit(1)
        self.apply_operator('MutationPopulation')
        if not self.check():
                print('MutationPopulation')
                exit(1)
        self.apply_operator("FitnessPopulation")
        if not self.check():
                print("FitnessPopulation")
                exit(1)
        self.apply_operator("Elitism")
        if not self.check():
                print("Elitism")
                exit(1)
        self.apply_operator('RestrictPopulation')
        if not self.check():
                print('RestrictPopulation')
                exit(1)


        # Optimizer for structure differencial Tokens (\/)
        # Lasso + Linear Regression for search amplitudes (\/)
        # Selection (\/)
        # Crossover (\/)
        # FitnessEvaluation (\/)
        # Elitism (\/)
        # Mutation (\/)
        # FitnessEvaluation (\/)
        # Restrict population

    def evolutionary(self, *args):  
        # self.apply_operator('InitPopulation')
        for n in range(self.iterations):
            # print('{}/{}\n'.format(n, self.iterations))
            self._evolutionary_step(args[0])
        set_constants(best_individ=list(filter(lambda ind: ind.elitism, self.structure))[0])

    
class Subpopulation(Population):
    def __init__(self, structure: list = None, iterations: int = 0, type_: str = "Subpopulation") -> None:
        super().__init__(structure)

        self.iterations = iterations
        self.type_ = type_

    def _evolutionary_step(self):
        for sub_population in self.structure:
            sub_population.evolutionary(self.structure)
        # self.apply_operator("DifferentialTokensOptimizerPopulation")

    def evolutionary(self):  
        self.apply_operator('InitSubPopulation')
        for n in range(self.iterations):
            print('{}/{}\n'.format(n, self.iterations))
            self._evolutionary_step()
