import numpy as np
import pandas as pd

from . import rules
from . import models

class DM:
    """ CLass for modelling the decision-making system. """
    def __init__(self, rules_json, model_list, data):
        """ Build a new decision-making system.

        Parameters
        ----------
        - rules_json: list of rules in json file, as described in README.md.
        - model_list: list of Model objects.
        - data: pandas dataframe containing input data, data.columns will be used as features.
        """

        assert isinstance(rules_json, dict), "Error: rules_json must be a dictionary!"
        assert isinstance(model_list, list), "Error: models must be a list of Model objects!"
        for i in range(len(model_list)):
            assert isinstance(model_list[i], models.Model), "Error: models must be a list of Model objects!"
        assert isinstance(data, pd.DataFrame), "Error: data must be a pandas DataFrame!"
        self.rules = {k: rules.Rule(rules_json, k) for k in rules_json.keys()}  # manage rules as Rule objects
        self.models = model_list
        self.data = data
        self.features = data.columns  # features used for explanation
        self.full_data = self.__run_models__(data)  # apply the models, complete data with their output
        self.variables = self.full_data.columns  # variables contains both input features and models output.

    def make_decision(self, example, verbose=False):
        if example.ndim > 1:
            decisions = [0 for i in range(len(example))]
            if not isinstance(example, pd.DataFrame):
                example = pd.DataFrame(example, columns=self.features)
            example = self.__run_models__(example)
            for i in range(len(example)):
                decisions[i] = self.__decide__(example.loc[i], verbose)
        else:
            if not isinstance(example, pd.DataFrame):
                example = pd.DataFrame(example, self.features).T
            example = self.__run_models__(example)
            decisions = [self.__decide__(example, verbose)]
        return np.array(decisions)

    def make_decision_eval(self, example):
        if example.ndim > 1:
            decisions = [0 for i in range(len(example))]
            if not isinstance(example, pd.DataFrame):
                example = pd.DataFrame(example, columns=self.features)
            example = self.__run_models__(example)
            for i in range(len(example)):
                decisions[i] = self.__decide_eval__(example.loc[i])
        else:
            if not isinstance(example, pd.DataFrame):
                example = pd.DataFrame(example, self.features).T
            example = self.__run_models__(example)
            decisions = [self.__decide_eval__(example)]
        return np.array(decisions)

    def __decide_eval__(self, example):
        # binary output, used for evaluation as it is required by other methods.
        for k, v in self.rules.items():
            z = [example[lbl] for lbl in v.actives]
            if (np.dot(v.A, z).T - v.b >= 0).all():
                decision = 1
            else:
                decision = 0
        return decision

    def make_decision_class(self, example):
        res = self.make_decision_eval(example)
        return np.array([1 - res, res]).T

    def __decide__(self, example, verbose):
        decisions = []
        triggered = []
        for k, v in self.rules.items():
            z = [example[lbl] for lbl in v.actives]
            if (np.dot(v.A, z).T - v.b >= 0).all():
                decisions.append(v.decision)
                triggered.append(k)
        if not decisions:
            decisions = None
            triggered = None
        if verbose:
            print(('Rule(s) ' + str(triggered) + ' triggered.') if triggered else 'No rules have been triggered.')
            print(('Decision(s) ' + str(decisions) + ' made.') if decisions else 'No decision made.')
        self.triggered = triggered
        return decisions

    def __run_models__(self, data):
        models = self.models
        full_data = data.copy()
        for model in models:
            full_data[model.name] = model.predict(data)
        return full_data
