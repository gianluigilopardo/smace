# Evaluation - Composite system with one linear model

import json
import logging
import os
import sys
import warnings

import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from smace import utils
from smace.decisions import DM
from smace.explainer import Smace

import utils as exp_utils

warnings.filterwarnings("ignore")
SEED = 0
np.random.seed(seed=SEED)

# path here
path = os.getcwd().replace(os.path.join("evaluation", "experiments"), "")
sys.path.append(path)


N_example = 100
N_sample = 1000
D = 6  # input features

# decision rule
rule_file = os.path.join(path, "evaluation", "experiments", "rules", "r01.json")
rule_name = "r01"

# input data
data = np.random.rand(N_sample, D)
df = pd.DataFrame(data)
df.columns = ["x" + str(i) for i in range(D)]
to = 1  # from 0 to 1
local = True
what = "rule_only_" + rule_name

# decision rules
with open(rule_file, "r") as fp:
    rules_json = json.load(fp)


# no models
models_list = []
N = len(models_list)

# decision system
dm = DM(rules_json, models_list, df)

# Initialize the explainers
explainer = Smace(dm)
data_summary = shap.sample(df, 100)
shap_explainer = shap.KernelExplainer(dm.make_decision_eval, data_summary)
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    df.values,
    feature_names=df.columns,
    discretize_continuous=True,
    verbose=True,
    mode="classification",
)
dec_avg = dm.make_decision_eval(df).mean()
print("Decision avg: ", dec_avg)

# examples to explain
random_example = df.copy()
example = random_example[dm.make_decision_eval(random_example) == 1 - to]
full_example = dm.__run_models__(example)
full_example["dist"] = 0
scale = dm.full_data.max() - dm.full_data.min()
for i, row in full_example.iterrows():
    full_example.dist.loc[i] = np.linalg.norm(
        (row[dm.rules[rule_name].variables] - dm.rules[rule_name].values) / scale, 2
    )
example = example.loc[full_example.sort_values("dist")[:N_example].index].reset_index(
    drop=True
)

# evaluation
smace_eval, lime_eval, shap_eval, random_eval = None, None, None, None
for i, xi in example.iterrows():
    print("\n", what, " > i: ", i)
    print(xi)
    smace_exp = explainer.explain(xi, rule_name)
    explanation = smace_exp.exp
    shap_values = shap_explainer.shap_values(xi)
    lime_values = utils.lime_mapper(
        lime_explainer.explain_instance(xi, dm.make_decision_class, num_features=D)
    )
    e_rule = smace_exp.rule_table(D + N)
    exp = pd.DataFrame(index=list(explanation.keys()))
    exp["SMACE"] = list(explanation.values())
    exp["SHAP"] = shap_values
    exp["LIME"] = lime_values
    print(exp)
    print(e_rule)
    smace_rank = exp.SMACE[exp.SMACE < 0].sort_values(ascending=True).index
    shap_rank = exp.SHAP[exp.SHAP < 0].sort_values(ascending=True).index
    lime_rank = exp.LIME[exp.LIME < 0].sort_values(ascending=True).index
    sample = exp_utils.perturb(xi, df, N_sample, dm, to, local=local)

    if smace_eval is not None:
        smace_eval = np.concatenate(
            (
                smace_eval,
                exp_utils.evaluate(to, xi, sample, smace_rank, dm, N_sample, df),
            )
        )
        lime_eval = np.concatenate(
            (lime_eval, exp_utils.evaluate(to, xi, sample, lime_rank, dm, N_sample, df))
        )
        shap_eval = np.concatenate(
            (shap_eval, exp_utils.evaluate(to, xi, sample, shap_rank, dm, N_sample, df))
        )
        random_eval = np.concatenate(
            (random_eval, exp_utils.evaluate(to, xi, sample, None, dm, N_sample, df))
        )
    else:
        smace_eval = exp_utils.evaluate(to, xi, sample, smace_rank, dm, N_sample, df)
        lime_eval = exp_utils.evaluate(to, xi, sample, lime_rank, dm, N_sample, df)
        shap_eval = exp_utils.evaluate(to, xi, sample, shap_rank, dm, N_sample, df)
        random_eval = exp_utils.evaluate(to, xi, sample, None, dm, N_sample, df)

eval_ = pd.DataFrame()
eval_["SMACE"] = smace_eval.mean(0)
eval_["SHAP"] = shap_eval.mean(0)
eval_["LIME"] = lime_eval.mean(0)
eval_["random"] = random_eval.mean(0)
print(eval_)

auc = 1 / 2 * (eval_.iloc[0] + 2 * eval_.iloc[1:-1].sum() + eval_.iloc[-1])
print(auc)

file = "results/" + what
with open(file + ".log", "w"):
    pass

eval_std = pd.DataFrame()
print(smace_eval)
eval_std["SMACE"] = smace_eval.std(0) / np.sqrt(N_sample)
eval_std["SHAP"] = shap_eval.std(0) / np.sqrt(N_sample)
eval_std["LIME"] = lime_eval.std(0) / np.sqrt(N_sample)
eval_std["random"] = random_eval.std(0) / np.sqrt(N_sample)

alpha = 0.2
fig, ax = plt.subplots()
ax.plot(eval_.LIME, color="tab:orange", label="LIME")
ax.fill_between(
    eval_.index,
    eval_.LIME - eval_std.LIME,
    eval_.LIME + eval_std.LIME,
    color="tab:orange",
    alpha=alpha,
)
ax.plot(eval_.SHAP, color="tab:green", label="SHAP")
ax.fill_between(
    eval_.index,
    eval_.SHAP - eval_std.SHAP,
    eval_.SHAP + eval_std.SHAP,
    color="tab:green",
    alpha=alpha,
)
ax.plot(eval_.random, color="tab:red", label="random")
ax.fill_between(
    eval_.index,
    eval_.random - eval_std.random,
    eval_.random + eval_std.random,
    color="tab:red",
    alpha=alpha,
)
ax.plot(eval_["SMACE"], color="tab:blue", label="SMACE")
ax.fill_between(
    eval_.index,
    eval_["SMACE"] - eval_std["SMACE"],
    eval_["SMACE"] + eval_std["SMACE"],
    color="tab:blue",
    alpha=alpha,
)
ax.legend()
plt.savefig(fname=str(file + ".pdf"))

# log
logging.basicConfig(
    format="%(message)s", filename=str(file) + ".log", level=logging.INFO
)
logging.info(what + "\n")
logging.info("RULE: " + str(dm.rules[rule_name].labels) + "\n")
logging.info("Decision avg: " + str(dec_avg) + "\n")
logging.info("Local sampling: " + str(local) + "\n")
logging.info("N_example: " + str(N_example) + "\n")
logging.info("N_sample: " + str(N_sample) + "\n")
logging.info("Evaluation: \n" + eval_.to_string() + "\n")
logging.info("Evaluation std: \n" + eval_std.to_string() + "\n")
logging.info("AUC: \n" + auc.to_string() + "\n")
