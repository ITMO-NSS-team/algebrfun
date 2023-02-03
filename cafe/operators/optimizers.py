from .base import GeneticOperatorIndivid
from .base import GeneticOperatorPopulation

class TokenParametersOptimizerIndivid(GeneticOperatorIndivid):
    """
    """

    def __init__(self, params=None):
        if params is None:
            params = {}
        add_params = {
            'optimizer': 'DE',
            'optimize_id': None,
            'popsize': 7,
            'eps': 0.005
        }
        for key, value in add_params.items():
            if key not in params.keys():
                params[key] = value
        super().__init__(params=params)
        # self._check_params('grid', 'optimizer', 'optimize_id', 'popsize', 'eps')

    

class TokenParametersOptimizerPopulation(GeneticOperatorPopulation):
    """
    """

    def __init__(self, params: dict = None):
        super().__init__(params=params)

    def apply(self, population, *args, **kwargs) -> None:
        for individ in population:
