import numpy as np
import pandas as pd


class Model:
    """ Each model within the decision system. """
    def __init__(self, model, name, df, mode='regression', preprocess=None):
        """ Define a new model, defined by a name.

        Parameters
        ----------
        - model: sklearn model or any object with a .predict and/or .predict_proba method, trained on df.
        - name: name of the model, the same used within the rules.
        - df: pandas dataframe, training set for model.
        - model: 'regression' or 'classification'. Only binary classification is supported.
        - preprocess: function to be applied to example if an intermediate preprocess is performed.
        """

        self.model = model
        self.name = name
        self.df = df
        self.mode = mode
        self.features = df.columns
        self.predictor = self.predict
        self.preprocess = preprocess

    def predict(self, example):
        if self.preprocess:
            example = self.preprocess(example)
        if isinstance(example, pd.DataFrame):
            x = example.values
        else:
            x = example
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        if self.mode == 'regression':
            return self.model.predict(x)
        elif self.mode == 'classification':
            return self.model.predict_proba(x)[:, 1]
