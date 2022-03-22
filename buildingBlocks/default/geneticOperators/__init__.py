"""
This package contains default implementations of different Genetic Operators.
For example Crossover, Mutation, FitnessEvaluator, Selectors etc. for work with individs and population.

Тут реализации всех генетических операторов, большинство работают с объектами inplace. Самая темная и сложная область
это Optimizers и ComplexOptimizers в котором заключена самая сложная и некрасиво написанная логика.
"""

# (TODO созать собирательные генетические операторы которые будут в параметрах содержать другие генетические операторы
#  и вызывать их по мере необходимости. Тем самым будут созданы сложные операторы с цепочкой действий, чтобы
#  пользователю было проще)
