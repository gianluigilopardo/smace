import pandas as pd
import numpy as np
import cvxpy as cp
import shap
import lime.lime_tabular

from . import utils
from . import smace_explanation
from . import decisions


class Smace:
    """ Explained object. """

    def __init__(self, dm):
        """ Build a new Explainer.

        Parameters
        ----------
        - dm: DM Object
        """

        assert isinstance(dm, decisions.DM), "Error: dm must be a DM Object!"
        self.dm = dm  # DM Object
        self.data = self.dm.full_data  # includes models output and input data

    def __rules_contribution__(self, example, rule_name):
        rule = self.dm.rules[rule_name]
        variables = list(self.dm.features)  # input features
        for model in self.dm.models:
            variables.append(model.name)  # running features
        values = {variable: 0 for variable in variables}  # initialize contribution as 0 for each input
        example = pd.DataFrame(example, self.dm.features).T
        example = self.dm.__run_models__(example)
        # models = self.dm.models
        # variables = rule.variables
        actives = rule.actives
        A = rule.A
        b = rule.b
        # n = len(b)
        m = len(actives)
        z = [float(example[lbl]) for lbl in actives]  # numerical values of example
        # Construct the optimization problem.
        x = cp.Variable([m])
        objective = cp.Minimize(cp.atoms.norm2(x - z))
        constraints = [A @ x - b >= 0]
        prob = cp.Problem(objective, constraints)
        # The optimal objective value is returned by `prob.solve()`.
        problem = (np.dot(A, z) - b)
        result = prob.solve()
        br = np.abs(np.dot(A, x.value) - b)
        nA = A.nonzero()[1]
        bx = []
        els = []  # save el of nA already seen
        for el in nA:
            if el not in els:
                bxj = np.min(br[np.where(nA == el)])
                bx.append(bxj)
                els.append(el)
        rule_scale = utils.__get_scale_factors__(self.data, rule.actives)
        r = (- np.abs(x.value - z) + bx) / rule_scale
        s = np.sign(r + np.finfo(float).eps)
        r = s * (1 - np.abs(r))
        r = np.round(r, 3)
        r_values = {feature: r[actives.index(feature)] for feature in actives}
        values.update(r_values)
        return values

    def __explain_model__(self, example, model, phi_model=None):
        input_features = list(self.dm.features)
        phi_values = {feature: 0 for feature in input_features}  # initialize importance as 0 for each input
        model_features = list(model.features)  # in general, model.features in input_features
        phi = 0
        if phi_model:
            phi_sum = np.sum(np.abs(list(phi_model.values())))
            phi_dict = {k: v / phi_sum for k, v in phi_model.items()}
            # phi_dict = {k: v / self.data[model.name].std() for k, v in phi_model.items()}
        else:  # shap
            data_summary = shap.sample(self.dm.data, 100)
            explainer = shap.KernelExplainer(model.predictor, data_summary)
            shap_values = explainer.shap_values(example)
            phi = shap_values
            phi = phi / np.sum(np.abs(shap_values))
            # phi = phi / self.data[model.name].std()
            phi_dict = {feature: phi[model_features.index(feature)] for feature in model_features}
        phi_values.update(phi_dict)
        return phi_values

    def explain(self, example, rule_name, phis=None):
        r = self.__rules_contribution__(example, rule_name)
        e = {feature: r[feature] for feature in self.dm.features}  # if no models
        phi = {}
        phi_model = None
        for model in self.dm.models:
            if model.name in self.dm.rules[rule_name].actives:
                if phis:
                    phi_model = phis[model.name]
                phi[model.name] = self.__explain_model__(example, model, phi_model)
                e = {feature: (e[feature] + r[model.name] * phi[model.name][feature]) for feature in
                     self.dm.features}
        explanation = smace_explanation.SmaceExplanation(example, e, r, phi)
        return explanation
