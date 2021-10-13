import numpy as np
import pandas as pd


class Model:
    def __init__(self, model, name, df, mode='regression', preprocess=None):
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
