import numpy as np

from . import utils


class Rule:
    def __init__(self, dict_rules, k):
        self.dict_rules = dict_rules
        self.name = k
        self.rule = dict_rules[k]
        self.conditions = self.rule['conditions']
        self.decision = self.rule['decision']
        self.variables, self.operators, self.values, self.labels = self.__map_rules__()
        indexes = np.unique(np.array(self.variables), return_index=True)[1]
        self.actives = [self.variables[index] for index in sorted(indexes)]
        self.A, self.b = self.__get_matrices__()

    def __map_rules__(self):
        variables = []
        operators = []
        values = []
        labels = []
        for condition in self.conditions:
            variable = condition['name']
            variables.append(variable)
            value = condition['value']
            values.append(value)
            operator = condition['operator']
            operators.append(operator)
            label = variable + utils.map_condition(operator) + str(value)
            labels.append(label)
        return variables, operators, values, labels

    def __get_matrices__(self):
        values = self.values
        variables = self.variables
        operators = self.operators
        actives = self.actives
        n = len(self.conditions)
        m = len(np.unique(variables))
        A = np.zeros([n, m])
        b = np.zeros(n)
        for i in range(n):
            operator = operators[i]
            value = values[i]
            variable = variables[i]
            j = actives.index(variable)
            A[i, j], b[i] = utils.map_constraints(operator, value)
        return A, b
