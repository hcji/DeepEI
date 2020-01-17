# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:25:43 2020

@author: hcji
"""

# draft
import pandas as pd
import matplotlib.pyplot as plt

mlp = pd.read_csv('Fingerprint/results/mlp_result.txt', sep='\t', header=None)
mlp.columns = ['id', 'accuracy', 'precision', 'recall', 'f1']
plsda = pd.read_csv('Fingerprint/results/plsda_result.txt', sep='\t', header=None)
plsda.columns = ['id', 'accuracy', 'precision', 'recall', 'f1']
lr = pd.read_csv('Fingerprint/results/lr_result.txt', sep='\t', header=None)
lr.columns = ['id', 'accuracy', 'precision', 'recall', 'f1']
xgb = pd.read_csv('Fingerprint/results/xgb_result.txt', sep='\t', header=None)
xgb.columns = ['id', 'accuracy', 'precision', 'recall', 'f1']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
axes[0, 0].violinplot( [mlp['accuracy'], plsda['accuracy'], lr['accuracy'], xgb['accuracy']] , showmeans=False, showmedians=True)
axes[0, 1].violinplot( [mlp['precision'], plsda['precision'], lr['precision'], xgb['precision']] , showmeans=False, showmedians=True)
axes[1, 0].violinplot( [mlp['recall'], plsda['recall'], lr['recall'], xgb['recall']] , showmeans=False, showmedians=True)
axes[1, 1].violinplot( [mlp['f1'], plsda['f1'], lr['f1'], xgb['f1']] , showmeans=False, showmedians=True)
axes[0, 0].set_ylabel('Accuracy')
axes[0, 1].set_ylabel('Precision')
axes[1, 0].set_ylabel('Recall')
axes[1, 1].set_ylabel('F1 Score')

plt.setp(axes, xticklabels=['', 'MLP', '', 'PLS-DA', '', 'LR', '', 'XGBoost'])
plt.show()
