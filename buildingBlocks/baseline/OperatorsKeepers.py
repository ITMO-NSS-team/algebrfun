"""
Contains Operators Map that maps names of genetic operators and their implementation.

Classes:
----------

OperatorsKeeper
"""
from buildingBlocks.baseline.BasicStructures import GeneticOperator


class OperatorsKeeper:
    """
    Dataclass using as dict.
    Mapping names of operators with their implementations.
    """

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        assert isinstance(value, GeneticOperator), 'Attribute must be "GeneticOperator" object'
        self.__dict__[key] = value














