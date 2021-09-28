import numpy as np
import pandas as pd


class Model:
    def __init__(self, model, name, df, mode='regression'):
        self.model = model
        self.name = name
        self.df = df
        self.mode = mode
        self.features = df.columns
        self.predictor = self.predict

    def predict(self, example):
        if isinstance(example, pd.DataFrame):
            x = example.values
        else:
            x = example
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        return self.model(x)
