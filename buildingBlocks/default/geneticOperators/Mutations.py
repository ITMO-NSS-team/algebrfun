import random
from multiprocessing import current_process
# from buildingBlocks.baseline.GeneticOperators import GeneticOperatorIndivid, GeneticOperatorPopulation
from buildingBlocks.baseline.BasicEvolutionaryEntities import GeneticOperatorIndivid, GeneticOperatorPopulation
from buildingBlocks.baseline.BasicEvolutionaryEntities import ComplexToken
from buildingBlocks.Globals.GlobalEntities import get_full_constant
import numpy as np

from buildingBlocks.default.geneticOperators.supplementary.Other import check_operators_from_kwargs, apply_decorator


# TODO Сделать отдельные операторы для работы с уравнениями и аппроксимацией ТС
class MutationIndivid(GeneticOperatorIndivid):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('mut_intensive', 'increase_prob', 'tokens')

    @apply_decorator
    def apply(self, individ, *args, **kwargs) -> None:
        # individ.kind += '->mutation'
        tokens = list(filter(lambda token: token not in individ.structure, self.params['tokens']))
        mandatory_tokens = list(filter(lambda token: token.mandatory != 0, tokens))
        non_mandatory_tokens = list(filter(lambda token: token.mandatory == 0, tokens))

        if not tokens:
            return
        # define how many tokens will be added
        # mut_intensive = np.random.randint(1, 1 + min(self.params['mut_intensive'],
        #                                              len(tokens)))
        # mut_intensive = np.random.randint(0, self.params['mut_intensive'])
        mut_intensive = self.params['mut_intensive']
        # if mut_intensive <= 0:
        #     return
        # Если длина структуры = 0 и обязательных токенов нет/1 и интенсивность мутации так вышло что 0,
        # То увеличиваем интенсивность мутации
        # if len(individ.structure) + len(mandatory_tokens) + mut_intensive <= 1:
        #     mut_intensive += 1

        # add_tokens = []
        # add_tokens.extend(mandatory_tokens)
        if mandatory_tokens:
            individ.add_substructure([token.clean_copy() for token in mandatory_tokens])

        add_tokens = []
        # add_tokens = list(np.random.choice(non_mandatory_tokens, size=mut_intensive, replace=True))

        # Тут возможен бесконечный цикл в случае набора необязательных токенов в виде разных констант или нет
        # (например если будут подаваться синусы и у них будет срабатывать __eq__
        # когда mut_intensive > чем их количества
        if mut_intensive > len(non_mandatory_tokens):
            add_tokens.extend(non_mandatory_tokens)
        else:
            for _ in range(mut_intensive):
                while True:
                    new_token = np.random.choice(non_mandatory_tokens).clean_copy()
                    if new_token not in add_tokens:
                        add_tokens.append(new_token)
                        break

        #TODO: добавить выборщик токена в зависимости от текущего вида индивида

        # add_tokens = [token.copy() for token in add_tokens]
        # for idx, token in enumerate(add_tokens):
        #     add_tokens[idx] = token.copy()
        # tokens is added to the chromo/ replace tokens in chromo/ both variants
        random.shuffle(add_tokens)

        # чем то резким заменить не получится ибо будет плохо оптимизироваться с учетом плохих токенов
        # а вот сменить стоящий токен возможно (плохо), плюс лассо освобождает места, плюс будет кроссовер для таких дел
        if individ.max_tokens > len(individ.structure) and np.random.uniform() <= self.params['increase_prob']:
            individ.add_substructure(add_tokens)
        elif individ.structure:
            idxs_to_choice = list(filter(lambda idx: individ.structure[idx].mandatory == 0, # and individ.structure no is CImp
                                         range(len(individ.structure))))
            if not idxs_to_choice:
                return
            probabilities = np.array(list(map(lambda idx: individ.structure[idx].fitness, idxs_to_choice)))
            probabilities /= probabilities.sum()
            for idx in np.random.choice(idxs_to_choice,
                                        size=min(len(idxs_to_choice), len(add_tokens)),
                                        replace=False,
                                        p=probabilities):
                individ.set_substructure(add_tokens.pop(), idx)
            # if len(add_functions) > len(chromo) we add token to the chromo
            if add_tokens:
                individ.add_substructure(add_tokens)

        # try:
        #     individ.forms.append(type(self).__name__ + individ.formula() + '<---' + current_process().name)
        # except:
        #     pass

class MutationIndividTerms(GeneticOperatorIndivid):
    def __init__(self, params) -> None:
        super().__init__(params=params)
        self._check_params('mut_intensive', 'increase_prob', 'tokens')

    @apply_decorator
    def apply(self, individ, *args, **kwargs) -> None:
        constants = get_full_constant()
        name_of_terms = list(map(lambda x: x.params[1]._name, individ.structure))
        # name_of_terms = np.array(name_of_terms)
        tokens = []
        for dftoken in constants['tokens']:
            if dftoken.params[1]._name in name_of_terms:
                continue
            tokens.append(dftoken)
        
        if len(tokens) == 0:
            return

        mut_intensive = self.params['mut_intensive']
        add_tokens = []
        add_tokens_names = []

        if mut_intensive > len(tokens):
            add_tokens.extend(tokens)
        else:
            for _ in range(mut_intensive):
                while True:
                    new_token = np.random.choice(tokens).copy()
                    if new_token.params[1]._name not in add_tokens_names:
                        add_tokens.append(new_token)
                        add_tokens_names.append(new_token.params[1]._name)
                        break
        
        random.shuffle(add_tokens)

        if individ.max_tokens > len(individ.structure) and np.random.uniform() <= self.params['increase_prob']:
            individ.add_substructure(add_tokens)
        elif individ.structure:
            idxs_to_choice = list(filter(lambda idx: individ.structure[idx].mandatory == 0, # and individ.structure no is CImp
                                         range(len(individ.structure))))
            if not idxs_to_choice:
                return
            probabilities = np.array(list(map(lambda idx: individ.structure[idx].fitness, idxs_to_choice)))
            probabilities /= probabilities.sum()
            for idx in np.random.choice(idxs_to_choice,
                                        size=min(len(idxs_to_choice), len(add_tokens)),
                                        replace=False,
                                        p=probabilities):
                individ.set_substructure(add_tokens.pop(), idx)
            # if len(add_functions) > len(chromo) we add token to the chromo
            if add_tokens:
                individ.add_substructure(add_tokens)




class ProductTokenMutationIndivid(GeneticOperatorIndivid):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('mut_prob', 'tokens', 'product_token', 'max_multi_len')

    @apply_decorator
    def apply(self, individ, *args, **kwargs) -> None:
        tokens = self.params['tokens']
        if len(tokens) == 0 or np.random.uniform() < self.params['mut_prob']:
            return
        add_token = np.random.choice(tokens,).copy()
        product_token = self.params['product_token'].copy()
        # idx = np.random.randint(0, len(ind.structure))
        idxs = [i for i in range(len(individ.structure))]
        random.shuffle(idxs)
        # TODO: make loop for add_tokens if len(add_tokens) > 1
        for idx in idxs:
            # if isinstance(individ.structure[idx], type(product_token)):
            if type(individ.structure[idx]) == type(product_token):
                ind_chromo_token_sub_len = len(individ.structure[idx].subtokens)
                if (ind_chromo_token_sub_len < self.params['max_multi_len']
                        and np.random.uniform() < 0.5):
                    individ.structure[idx].add_subtoken(add_token)
                    if individ.structure.count(individ.structure[idx]) > 1:
                        individ.structure[idx].del_subtoken(add_token)
                    else:
                        break
                else:
                    token_idxs = [i for i in range(ind_chromo_token_sub_len)]
                    random.shuffle(token_idxs)
                    for token_idx in token_idxs:
                        old_subtoken = individ.structure[idx].subtokens[token_idx]
                        individ.structure[idx].set_subtoken(add_token, idx=token_idx)
                        if individ.structure.count(individ.structure[idx]) > 1:
                            individ.structure[idx].set_subtoken(old_subtoken, idx=token_idx)
                            flag = False
                        else:
                            flag = True
                            break
                        if flag:
                            break
            elif individ.structure[idx].name_ in list(map(lambda token: token.name_,
                                                          tokens)) and not isinstance(individ.structure[idx],
                                                                                      ComplexToken):
                # если токен не является сложным ( в частности продуктом), но в списке простых токенов
                if self.params['max_multi_len'] <= 1:
                    return
                new_product_token = product_token
                new_product_token.subtokens = [individ.structure[idx], add_token]
                if individ.structure.count(new_product_token) == 0:
                    individ.structure[idx] = new_product_token
                    assert len(individ.structure[idx].subtokens) <= self.params['max_multi_len']
                    break
        individ.change_all_fixes(False)


class ImpComplexMutationIndivid(GeneticOperatorIndivid):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('mut_prob', 'complex_token', 'grid', 'threshold')

    @apply_decorator
    def apply(self, individ, *args, **kwargs) -> None:
        complex_token = self.params['complex_token']
        choiced_tokens = list(filter(lambda token: type(token) == type(complex_token.pattern) and
                                     token.fixator['self'],
                                     individ.structure))
        if len(choiced_tokens) == 0:
            return

        if np.random.uniform() > self.params['mut_prob']:
            return

        # del non optimized pulses
        # for token in individ.structure:
        #     if type(token) == type(complex_token.pattern) and not token.fixator['self']:
        #         individ.del_substructure(token)

        grid = self.params['grid']
        all_mx = []
        for idx in range(grid.shape[0]):
            step = np.mean(grid[idx][1:] - grid[idx][:-1])
            all_mx.append(np.fft.fftfreq(len(grid[idx]), step).max())
        
        # wmax = np.max(all_mx)

        threshold = self.params['threshold']
        for idx, token in enumerate(individ.structure):
            if token in choiced_tokens:
                if np.any(token.param('Frequency') > threshold*np.array(all_mx)):
                    continue
                new_complex_token = complex_token.extra_clean_copy()
                new_complex_token.pattern = token.copy()
                individ.set_substructure(new_complex_token, idx=idx)
                break
        # превращаем только один импульс, вклад которого максимален и он не слишком высокочастотный


class MutationPopulation(GeneticOperatorPopulation):
    def __init__(self, params):
        super().__init__(params=params)
        self._check_params('mutation_size')

    def apply(self, population, *args, **kwargs):
        selected_population = list(filter(lambda individ: individ.selected, population.structure))
        mutation_size = self.params['mutation_size']
        if mutation_size is None or mutation_size > len(selected_population):
            selected_individs = selected_population
        else:
            # assert mutation_size <= len(selected_population), "Mutations size must be less than population size"
            selected_individs = np.random.choice(selected_population, replace=False, size=mutation_size)

        for individ in selected_individs:
            if individ.elitism:
                individ.elitism = False
                new_individ = individ.copy()
                new_individ.selected = False
                population.structure.append(new_individ)
            if individ.type_ == "DEquation":
                individ.apply_operator('MutationIndividTerms')
            else:
                individ.apply_operator('MutationIndivid')
                individ.apply_operator('ImpComplexMutationIndivid')
            # individ.apply_operator('ProductTokenMutationIndivid')
        return population

