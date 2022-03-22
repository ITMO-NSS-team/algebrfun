

def objective1(individ):
    individ.apply_operator('VarFitnessIndivid')
    return individ.fitness


def objective2(individ):
    return len(individ.structure)
