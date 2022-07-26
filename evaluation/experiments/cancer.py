# Evaluation - Breast Cancer

import json
import logging
import os
import pickle
import sys
import warnings

import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
SEED = 0
np.random.seed(seed=SEED)

# path here
path = os.getcwd().replace('evaluation\\experiments', '')
sys.path.append(path)

from smace import utils
# SMACE
from smace.decisions import DM
from smace.explainer import Smace
from smace.models import Model

# experiments
import utils as exp_utils

N_example = 100
N_sample = 1000
to = 1
rule_name = 'paper'
local = True

what = "cancer_" + rule_name

# input data
df = pd.read_csv('cancer_data.csv').drop(columns=['id', 'Unnamed: 32']).reset_index(drop=True)  # original data is huge

# decision rules
with open('rules/cancer_rule.json', 'r') as fp:
    rules_json = json.load(fp)

# preprocess
y = []
for row in df.diagnosis:
    if row == 'M':
        y.append(1)
    else:
        y.append(0)

data = df.drop(columns=['diagnosis'])
X = data.copy()

# models
mod = LogisticRegression().fit(X, y)


cancer_mod = Model(mod, 'cancer_risk', data, mode='classification')

models_list = [cancer_mod]

# decision system
dm = DM(rules_json, models_list, data)

# Initialize the explainers
explainer = Smace(dm)
data_summary = shap.sample(data, 100)
shap_explainer = shap.KernelExplainer(dm.make_decision_eval, data_summary)
lime_explainer = lime.lime_tabular.LimeTabularExplainer(data.values, feature_names=data.columns,
                                                        discretize_continuous=True, verbose=True,
                                                        mode='classification')

dec_avg = dm.make_decision_eval(data).mean()
print('Decision avg: ', dec_avg)

D = len(data.columns)
N = len(models_list)
print('D: ', str(D))
print('N: ', str(N))

# examples to explain
random_example = data.copy()
example = random_example[dm.make_decision_eval(random_example) == 1 - to]
full_example = dm.__run_models__(example)
full_example['dist'] = 0
scale = dm.full_data.max() - dm.full_data.min()
for i, row in full_example.iterrows():
    full_example.dist.loc[i] = np.linalg.norm((row[dm.rules[rule_name].variables] - dm.rules[rule_name].values) / scale,
                                              2)
example = example.loc[full_example.sort_values('dist')[:N_example].index].reset_index(drop=True)

# evaluation
smace_eval, lime_eval, shap_eval, random_eval = None, None, None, None
for i, xi in example.iterrows():
    print('\n', what, ' > i: ', i)
    print(xi)
    print(cancer_mod.predict(xi))
    smace_exp = explainer.explain(xi, rule_name)
    explanation = smace_exp.exp
    shap_values = shap_explainer.shap_values(xi)
    lime_values = utils.lime_mapper(lime_explainer.explain_instance(xi, dm.make_decision_class, num_features=D))
    e_rule = smace_exp.rule_table(D + N)
    exp = pd.DataFrame(index=list(explanation.keys()))
    exp['SMACE'] = list(explanation.values())
    exp['SHAP'] = shap_values
    exp['LIME'] = lime_values
    print(exp)
    print(e_rule)
    print(smace_exp.model_table('cancer_risk'))
    smace_rank = exp.SMACE[exp.SMACE < 0].sort_values(ascending=True).index
    shap_rank = exp.SHAP[exp.SHAP < 0].sort_values(ascending=True).index
    lime_rank = exp.LIME[exp.LIME < 0].sort_values(ascending=True).index
    sample = exp_utils.perturb(xi, data, N_sample, dm, to, local=local)

    if smace_eval is not None:
        smace_eval = np.concatenate((smace_eval, exp_utils.evaluate(to, xi, sample, smace_rank, dm, N_sample, data)))
        lime_eval = np.concatenate((lime_eval, exp_utils.evaluate(to, xi, sample, lime_rank, dm, N_sample, data)))
        shap_eval = np.concatenate((shap_eval, exp_utils.evaluate(to, xi, sample, shap_rank, dm, N_sample, data)))
        random_eval = np.concatenate((random_eval, exp_utils.evaluate(to, xi, sample, None, dm, N_sample, data)))
    else:
        smace_eval = exp_utils.evaluate(to, xi, sample, smace_rank, dm, N_sample, data)
        lime_eval = exp_utils.evaluate(to, xi, sample, lime_rank, dm, N_sample, data)
        shap_eval = exp_utils.evaluate(to, xi, sample, shap_rank, dm, N_sample, data)
        random_eval = exp_utils.evaluate(to, xi, sample, None, dm, N_sample, data)

eval_ = pd.DataFrame()
eval_['SMACE'] = smace_eval.mean(0)
eval_['SHAP'] = shap_eval.mean(0)
eval_['LIME'] = lime_eval.mean(0)
eval_['random'] = random_eval.mean(0)
print(eval_)

auc = 1 / 2 * (eval_.iloc[0] + 2 * eval_.iloc[1:-1].sum() + eval_.iloc[-1])
print(auc)

file = "results/" + what
with open(file + '.log', 'w'):
    pass

eval_std = pd.DataFrame()
print(smace_eval)
eval_std['SMACE'] = smace_eval.std(0) / np.sqrt(N_sample)
eval_std['SHAP'] = shap_eval.std(0) / np.sqrt(N_sample)
eval_std['LIME'] = lime_eval.std(0) / np.sqrt(N_sample)
eval_std['random'] = random_eval.std(0) / np.sqrt(N_sample)

res = {'eval': eval_, 'error': eval_std}
pickle.dump(res, open(file + '.p', 'wb'))

# log
logging.basicConfig(format='%(message)s', filename=str(file) + ".log", level=logging.INFO)
logging.info(what + '\n')
logging.info('RULE: ' + str(dm.rules[rule_name].labels) + '\n')
logging.info('Decision avg: ' + str(dec_avg) + '\n')
logging.info('Local sampling: ' + str(local) + '\n')
logging.info('N_example: ' + str(N_example) + '\n')
logging.info('N_sample: ' + str(N_sample) + '\n')
logging.info('Evaluation: \n' + eval_.to_string() + '\n')
logging.info('Evaluation std: \n' + eval_std.to_string() + '\n')
logging.info('AUC: \n' + auc.to_string() + '\n')
