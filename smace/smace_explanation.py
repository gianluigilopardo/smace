"""
smace_explanation docstring
"""


import pandas as pd


class SmaceExplanation:
    """ Object returned by explainers. """

    def __init__(self, example, exp, r, phi):
        """ Create a new Explanation.

        Parameters
        ----------
        - example: instance to be explained.
        - exp: overall contributions.
        - r: rule contributions
        - phi: contributions for models
        """

        self.example = example
        self.exp = exp
        self.rule = r
        self.models = phi

    def table(self, num_features=5):
        # return dataframe with overall contributions
        e = self.exp
        xi = self.example
        exp_table = pd.DataFrame(index=list(e.keys()))
        exp_table['Example'] = list(xi)
        exp_table['Contribution'] = list(e.values())
        exp_table = pd.DataFrame(exp_table)
        return exp_table.reindex(exp_table.Contribution.abs().sort_values(ascending=False).index)[:num_features]

    def bar(self, num_features=5):
        # return bar plot with overall contributions
        return self.table().Contribution.reindex(self.table().Contribution.abs().
                                                 sort_values(ascending=True).index)[-num_features:].plot.barh()

    def rule_table(self, num_features=5):
        # return dataframe with rule contributions
        e = self.rule
        exp_table = pd.DataFrame(index=list(e.keys()))
        exp_table['Contribution'] = list(e.values())
        exp_table = pd.DataFrame(exp_table)
        return exp_table.reindex(exp_table.Contribution.abs().sort_values(ascending=False).index)[:num_features]

    def rule_bar(self, num_features=5):
        # return bar plot with rule contributions
        return self.rule_table().Contribution.reindex(self.table().Contribution.abs().
                                                      sort_values(ascending=True).index)[-num_features:].plot.barh()

    def model_table(self, model_name, num_features=5):
        # return dataframe with model contributions
        e = self.models[model_name]
        exp_table = pd.DataFrame(index=list(e.keys()))
        exp_table['Contribution'] = list(e.values())
        exp_table = pd.DataFrame(exp_table)
        return exp_table.reindex(exp_table.Contribution.abs().sort_values(ascending=False).index)[:num_features]

    def model_bar(self, model_name, num_features=15):
        # return bar plot with model contributions
        return self.model_table(model_name).Contribution.reindex(self.table().Contribution.abs().
                                                                 sort_values(ascending=True).index)[
                                                                     -num_features:].plot.barh()
