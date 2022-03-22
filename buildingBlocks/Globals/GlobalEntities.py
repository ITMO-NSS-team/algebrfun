"""Глобальные сущности и константы, для удобства и доступа различных процессов к глобальным данным при параллелизации"""

global operators_mapper


def set_operators(operators):
    global operators_mapper
    operators_mapper = operators


def get_operators():
    return operators_mapper


global constants
constants = {}


def set_full_constant(const: dict) -> None:
    global constants
    constants = const


def get_full_constant() -> dict:
    return constants


def set_constants(**kwargs) -> None:
    for key, value in kwargs.items():
        constants[key] = value


def del_constants(*args) -> None:
    for key in constants.keys():
        if key in args:
            constants.pop(key)


global n_jobs


def get_n_jobs() -> int:
    return n_jobs


def set_n_jobs(n: int = 0) -> None:
    global n_jobs
    n_jobs = n


# Return all
def get_all_globals() -> dict:
    ret = {
        'operators': operators_mapper,
        'constants': constants,
        'n_jobs': n_jobs
    }
    return ret

