"""
utils docstring
"""


import numpy as np


def map_condition(operator):
    if operator == 'gt':
        condition = ' > '
    elif operator == 'geq':
        condition = ' >= '
    elif operator == 'leq':
        condition = ' <= '
    elif operator == 'lt':
        condition = ' < '
    return condition


def map_constraints(operator, value):
    eps = 1e-14
    a = 0
    b = 0
    if operator == 'gt':
        a, b = 1 + eps, value
    elif operator == 'geq':
        a, b = 1, value
    elif operator == 'leq':
        a, b = -1, -value
    elif operator == 'lt':
        a, b = -(1 + eps), -value
    return a, b


def lime_mapper(x):  # mapper needed for lime
    y = []
    x = x.as_map()[1]
    x.sort()
    y = [x[i][1] for i in range(len(x))]
    return np.array(y)


def __get_scale_factors__(data, variables):
    return [(data[variable].max() - data[variable].min()) for variable in variables]
