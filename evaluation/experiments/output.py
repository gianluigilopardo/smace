import pickle
import os

import seaborn as sns
from matplotlib import pyplot as plt

# specific parameters
sns.set_theme(style='darkgrid')
from matplotlib.ticker import MaxNLocator

# name = 'telco_paper'
# title = 'Retention Offer'
name = 'fraud_paper'
title = 'Fraud Detection'
# name = 'cancer_paper'
# title = 'Cancer Treatment'

res_path = os.path.join(os.getcwd(), 'results')

res = pickle.load(open(os.path.join(res_path, name + '.p'), 'rb'))

data = res['eval']
error = res['error']

# data order: SMACE, SHAP, LIME, random
alpha = 0.2
lw = 3  # linewidth
fs = 20  # fontsize
sns.set(font_scale=1.5)
p = sns.lineplot()

p.set_xlabel('number of removed features', fontsize=fs)
p.set_ylabel('average decision', fontsize=fs)

plt.plot(data['SMACE'], color='tab:blue', label='SMACE', linewidth=lw)
plt.fill_between(data.index, data['SMACE'] - error['SMACE'], data['SMACE'] + error['SMACE'], color='tab:blue',
                 alpha=alpha)

plt.plot(data.SHAP, color='tab:green', label='SHAP', linewidth=lw)
plt.fill_between(data.index, data.SHAP - error.SHAP, data.SHAP + error.SHAP, color='tab:green', alpha=alpha)
plt.plot(data.LIME, color='tab:orange', label='LIME', linewidth=lw)
plt.fill_between(data.index, data.LIME - error.LIME, data.LIME + error.LIME, color='tab:orange', alpha=alpha)
plt.plot(data.random, color='tab:red', label='random', linewidth=lw)
plt.fill_between(data.index, data.random - error.random, data.random + error.random, color='tab:red', alpha=alpha)
p.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.title(title)
plt.legend(fontsize=fs)
plt.plot(data['SMACE'], color='tab:blue', label='SMACE', linewidth=lw)
plt.fill_between(data.index, data['SMACE'] - error['SMACE'], data['SMACE'] + error['SMACE'], color='tab:blue',
                 alpha=alpha)
filename = os.path.join(res_path, name)
plt.savefig(fname=str(filename + '.pdf'), bbox_inches='tight', pad_inches=0)

