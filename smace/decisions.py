import numpy as np
import pandas as pd

from . import rules


class DM:
    def __init__(self, rules_json, models, data):
        self.rules = {k: rules.Rule(rules_json, k) for k in rules_json.keys()}
        self.models = models
        self.data = data
        self.features = data.columns
        self.full_data = self.__run_models__(data)
        self.variables = self.full_data.columns

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
            # if model.mode == 'regression':  # and model.name in self.rules.actives:
            #     full_data[model.name] = model.predict(data)
            # elif model.mode == 'classification':  # and model.name in self.rules.actives:
            #     full_data[model.name] = model.predict_proba(data)[:, 1]
        return full_data
