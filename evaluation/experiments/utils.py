"""
utils docstring
"""


import random

import numpy as np
import pandas as pd
from scipy.stats import truncnorm


# result
def evaluate(to, xi, sample, rank, dm, N_sample, df):
    D = len(xi)
    evals = np.array([[1 - to for j in range(D + 1)] for i in range(N_sample)])
    xi = pd.DataFrame([list(xi)], index=range(N_sample), columns=df.columns)
    if rank is None:  # random ranking
        random_rank = list(df.columns)
        n = random.randint(0, len(random_rank))
        random.shuffle(random_rank)
        rank = random_rank[:n]
    for j in range(D):
        if j < len(rank):
            xi[:][rank[j]] = sample[:][rank[j]]
        evals[:, j + 1] = dm.make_decision_eval(xi)
    return evals


def perturb(xi, df, N_sample, dm, to, local=True, categorical_names=None):
    scale = df.std()
    a, b = (df.min() - xi) / scale, (df.max() - xi) / scale
    if local:
        print(a)
        print(b)
        sample = truncnorm.rvs(a, b, loc=xi, scale=scale, size=[N_sample, len(xi)])
        sample = pd.DataFrame(sample)
        sample.columns = df.columns
        if categorical_names:
            sample[categorical_names] = (
                df[categorical_names]
                .sample(N_sample, replace=True)
                .reset_index(drop=True)
            )
    else:
        # sample = np.random.rand(N_sample, len(xi))
        sample = df.sample(N_sample, replace=True).reset_index(drop=True)
    return sample
