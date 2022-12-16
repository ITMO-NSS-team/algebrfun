from copy import deepcopy
from multiprocessing import current_process

from buildingBlocks.Globals.GlobalEntities import set_constants, get_full_constant


def check_or_create_fixator_item(individ, key: str) -> bool:
    try:
        fix = individ.fixator[key]
    except KeyError:
        individ.fixator[key] = False
        fix = False
    return fix


def apply_decorator(method):
    def wrapper(*args, **kwargs):
        self = args[0]
        try:
            individ = kwargs['individ']
        except KeyError:
            individ = args[1]

        # copy_individ = deepcopy(individ)
        ind_formula = deepcopy(individ.formula())

        fix = check_or_create_fixator_item(individ, type(self).__name__)
        if fix:
            return

        ret = method(*args, **kwargs)
        individ.fixator[type(self).__name__] = True

        try:
            if individ.formula() != ind_formula:
                individ.forms.append(type(self).__name__ + ': by {}\n'.format(current_process().name) + '---->'
                                     + individ.formula())
        except:
            pass

        return ret
    return wrapper


def check_operators_from_kwargs(**kwargs):
    try:
        operators = kwargs['operators']
        return operators
    except KeyError:
        raise KeyError('Not found arg "operators" in kwargs')


count = 0


def create_tmp_individ(individ, optimizing_tokens: list, target, name: str = 'tmp_target'):
    global count
    

    # target_tokens = list(filter(lambda token: token.mandatory != 0, individ.structure))
    # assert len(target_tokens) == 1, 'There must be only one target token'
    # target_token = target_tokens[0]

    # tmp_token = target_token.extra_clean_copy()
    # tmp_token.name_ = name
    # count += 1
    # set_constants(tmp2_target=target)
    constants = get_full_constant()
    constants[name] = target
    # !очень важно, чтобы не блокировались при вычислении
    # for token in optimizing_tokens:
    #     token.fixator['self'] = True

    tmp_individ = individ.clean_copy()
    tmp_individ.structure = optimizing_tokens
    if individ.type_ == "DEquation":
        target_tokens = list(filter(lambda token: token.mandatory != 0, individ.structure))
        assert len(target_tokens) == 1, 'There must be only one target token'
        target_token = target_tokens[0]
        tmp_token = target_token.extra_clean_copy()
        tmp_token.name_ = name
        tmp_individ.add_substructure(tmp_token)

    # tmp_individ.add_substructure(tmp_token)
    return tmp_individ
