import pandas as pd


class SmaceExplanation:
    """Object returned by explainers"""
    def __init__(self, example, exp, r, phi):
        # dictionaries for overall, rule, and models contributions
        self.example = example
        self.exp = exp
        self.rule = r
        self.models = phi

    def table(self, num_features=15):
        # return dataframe with overall contributions
        e = self.exp
        xi = self.example
        exp_table = pd.DataFrame(index=list(e.keys()))
        exp_table['Example'] = list(xi)
        exp_table['Contribution'] = list(e.values())
        exp_table = pd.DataFrame(exp_table)
        return exp_table.reindex(exp_table.Contribution.abs().sort_values(ascending=False).index)[:num_features]

    def bar(self, num_features=15):
        # return bar plot with overall contributions
        return self.table(num_features).Contribution.plot.barh()

    def rule_table(self, num_features=15):
        # return dataframe with rule contributions
        e = self.rule
        exp_table = pd.DataFrame(index=list(e.keys()))
        exp_table['Contribution'] = list(e.values())
        exp_table = pd.DataFrame(exp_table)
        return exp_table.reindex(exp_table.Contribution.abs().sort_values(ascending=False).index)[:num_features]

    def rule_bar(self, num_features=15):
        # return bar plot with rule contributions
        return self.rule_table(num_features).plot.barh()

    def model_table(self, model_name, num_features=15):
        # return dataframe with model contributions
        e = self.models[model_name]
        exp_table = pd.DataFrame(index=list(e.keys()))
        exp_table['Contribution'] = list(e.values())
        exp_table = pd.DataFrame(exp_table)
        return exp_table.reindex(exp_table.Contribution.abs().sort_values(ascending=False).index)[:num_features]

    def model_bar(self, model_name, num_features=15):
        # return bar plot with model contributions
        return self.model_table(model_name, num_features).plot.barh()
