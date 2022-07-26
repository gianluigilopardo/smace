"""
rules docstring
"""


import numpy as np

from . import utils


class Rule:
    """Each rule is a Rule object, made of conditions."""

    def __init__(self, dict_rules, k):
        """Build a new rule, defined by a name, a list of conditions, and a decision.

        Parameters
        ----------
        - dict_rules: list of rules in json file, as described in README.md.
        - k: name of the rule to handle, key of the dictionary.
        """

        assert isinstance(dict_rules[k], dict), (
            "Error: the rule " + k + " is not in the right format!"
        )
        assert "conditions" in dict_rules[k], (
            "Error: the rule " + k + " does not have the key 'conditions'!"
        )
        assert "decision" in dict_rules[k], (
            "Error: the rule " + k + " does not have the key 'decision'!"
        )
        self.dict_rules = dict_rules
        self.name = k
        self.rule = dict_rules[k]
        self.conditions = self.rule["conditions"]
        self.decision = self.rule["decision"]  # output if conditions are satisfied
        self.variables, self.operators, self.values, self.labels = self.__map_rules__()
        indexes = np.unique(np.array(self.variables), return_index=True)[1]
        self.actives = [
            self.variables[index] for index in sorted(indexes)
        ]  # list of variables directly involved
        (
            self.A,
            self.b,
        ) = self.__get_matrices__()  # conditions converted to matrix system

    def __map_rules__(self):
        variables = []
        operators = []
        values = []
        labels = []
        for condition in self.conditions:
            assert "name" in condition, (
                "Error: " + self.name + " conditions are missing the key 'name'!"
            )
            assert "value" in condition, (
                "Error: " + self.name + " conditions are missing the key 'value'!"
            )
            assert "operator" in condition, (
                "Error: " + self.name + " conditions are missing the key 'operator'!"
            )
            variable = condition["name"]
            variables.append(variable)
            value = condition["value"]
            values.append(value)
            operator = condition["operator"]
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
