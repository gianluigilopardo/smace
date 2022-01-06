# Evaluation - Telco use case

import numpy as np
import pandas as pd
import json
import warnings
import sys
import os
import logging
import matplotlib.pyplot as plt

import shap
import lime.lime_tabular

import xgboost as xgb
import sklearn

warnings.filterwarnings("ignore")
SEED = 0
np.random.seed(seed=SEED)

# path here
path = os.getcwd().replace('evaluation\\experiments', '')
sys.path.append(path)

# SMACE
from smace.decisions import DM
from smace.explainer import Smace
from smace.models import Model
from smace import utils

# experiments
import utils as exp_utils

N_example = 100
N_sample = 1000
to = 1
rule_name = 'simple'
local = True

what = "telco/" + rule_name + "_" + str(SEED)

# input data
df = pd.read_csv('telco_data.csv').drop(columns={'ID'})

# decision rules
with open('rules/telco_rules.json', 'r') as fp:
    rules_json = json.load(fp)

# preprocess
categorical_names = ['Gender', 'Status', 'Car Owner', 'Paymethod', 'LocalBilltype', 'LongDistanceBilltype']


def df_prep(dataframe):
    # String to numbers: {F,M} -> {0,1}
    for feature in categorical_names:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(dataframe[feature].astype(str))
        dataframe[feature] = le.transform(dataframe[feature].astype(str))
    return dataframe


# training data
y_cr = df.CHURN
y_ltv = df.LTV

data = df_prep(df.drop(columns={'CHURN', 'LTV'}))
categorical_features = []
for cat in categorical_names:
    categorical_features.append(list(data.columns).index(cat))

# preprocess
X = data.copy()
for feature in categorical_names:
    # ONE HOT ENCODING
    # Adding the new columns
    X = pd.concat([X, pd.get_dummies(X[feature], prefix=feature)], axis=1)
    # Removing the old nominal variables
    X.drop([feature], axis=1, inplace=True)
X = X.values

# models
xgb_cr = xgb.XGBClassifier(objective='reg:logistic').fit(X, y_cr)
xgb_ltv = xgb.XGBRegressor().fit(X, y_ltv)


# preprocess for example
def preprocess(x):
    X = data.copy()
    if x.ndim == 1:
        x = np.expand_dims(list(x), axis=0)
    x = pd.DataFrame(x, columns=list(X.columns))
    x = df_prep(x)
    X = X.append(x)
    for feature in categorical_names:
        # ONE HOT ENCODING
        # Adding the new columns
        X = pd.concat([X, pd.get_dummies(X[feature], prefix=feature)], axis=1)
        # Removing the old nominal variables
        X.drop([feature], axis=1, inplace=True)
    return X.tail(x.shape[0]).values.astype(np.float)


cr_mod = Model(xgb_cr, 'cr', data, mode='classification', preprocess=preprocess)
ltv_mod = Model(xgb_ltv, 'ltv', data, mode='regression', preprocess=preprocess)

models_list = [cr_mod, ltv_mod]

# decision system
dm = DM(rules_json, models_list, data)


# Initialize the explainers
explainer = Smace(dm)
data_summary = shap.sample(data, 100)
shap_explainer = shap.KernelExplainer(dm.make_decision_eval, data_summary)
lime_explainer = lime.lime_tabular.LimeTabularExplainer(data.values, feature_names=data.columns,
                                                        discretize_continuous=True, verbose=True,
                                                        mode='classification',
                                                        categorical_names=categorical_names,
                                                        categorical_features=categorical_features)

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
scale = dm.full_data.max()-dm.full_data.min()
for i, row in full_example.iterrows():
    full_example.dist.loc[i] = np.linalg.norm((row[dm.rules[rule_name].variables] - dm.rules[rule_name].values) / scale, 2)
example = example.loc[full_example.sort_values('dist')[:N_example].index].reset_index(drop=True)

# evaluation
smace_eval, lime_eval, shap_eval, random_eval = None, None, None, None
for i, xi in example.iterrows():
    print('\n', what, ' > i: ', i)
    print(xi)
    print(cr_mod.predict(xi))
    print(ltv_mod.predict(xi))
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
    print(smace_exp.model_table('cr'))
    print(smace_exp.model_table('ltv'))
    smace_rank = exp.SMACE[exp.SMACE < 0].sort_values(ascending=True).index
    shap_rank = exp.SHAP[exp.SHAP < 0].sort_values(ascending=True).index
    lime_rank = exp.LIME[exp.LIME < 0].sort_values(ascending=True).index
    sample = exp_utils.perturb(xi, data, N_sample, dm, to, local=local, categorical_names=categorical_names)

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
    print('SMACE: ', smace_rank, '- ', smace_eval.mean(0))
    print('SHAP:', shap_rank, '- ', shap_eval.mean(0))
    print('LIME: ', lime_rank, '- ', lime_eval.mean(0))

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

alpha = 0.2
fig, ax = plt.subplots()
ax.plot(eval_.LIME, color='tab:orange', label='LIME')
ax.fill_between(eval_.index, eval_.LIME - eval_std.LIME, eval_.LIME + eval_std.LIME, color='tab:orange', alpha=alpha)
ax.plot(eval_.SHAP, color='tab:green', label='SHAP')
ax.fill_between(eval_.index, eval_.SHAP - eval_std.SHAP, eval_.SHAP + eval_std.SHAP, color='tab:green', alpha=alpha)
ax.plot(eval_.random, color='tab:red', label='random')
ax.fill_between(eval_.index, eval_.random - eval_std.random, eval_.random + eval_std.random, color='tab:red',
                alpha=alpha)
ax.plot(eval_['SMACE'], color='tab:blue', label='SMACE')
ax.fill_between(eval_.index, eval_['SMACE'] - eval_std['SMACE'], eval_['SMACE'] + eval_std['SMACE'], color='tab:blue',
                alpha=alpha)
ax.legend()
plt.savefig(fname=str(file + '.pdf'))

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
