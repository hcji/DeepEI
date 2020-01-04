# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 13:31:51 2020

@author: hcji
"""


import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import load_npz, csr_matrix

from Fingerprint.mlp import MLP
from Fingerprint.cnn import CNN
from Fingerprint.plsda import PLSDA
from Fingerprint.lr import LR
from Fingerprint.xgb import XGBoost

# load data
with open('Data/split.json', 'r') as js:
    keep = np.array(json.load(js)['keep'])
spec = load_npz('Data/Peak_data.npz')
fps = load_npz('Data/CDK_fp.npz')

# exclude isolate test data
spec = spec.todense()[keep,:]
fps = csr_matrix(fps)[keep,:]

# build model
mlp_result = open('Fingerprint/results/mlp_result.txt', 'a+')
cnn_result = open('Fingerprint/results/cnn_result.txt', 'a+')
lr_result = open('Fingerprint/results/lr_result.txt', 'a+')
plsda_result = open('Fingerprint/results/plsda_result.txt', 'a+')
xgb_result = open('Fingerprint/results/xgb_result.txt', 'a+')

for i in tqdm(range(fps.shape[1])):
    y = fps[:,i].todense()
    y = np.squeeze(np.asarray(y))
    # check: 0.1 < bias < 0.9
    fr = np.sum(y) / len(y)
    if (fr < 0.1) or (fr > 0.9):
        continue
    Y = np.hstack((y, (1-y)))
    
    # mlp model
    mlp = MLP(spec, Y)
    mlp.train()
    mlp_res = mlp.test()
    mlp_result.write("\t".join([str(i) for i in mlp_res]))
    mlp.save('Fingerprint/mlp_models/{}.h5'.format(i))
    
    # cnn model
    cnn = CNN(spec, Y)
    cnn.train()
    cnn_res = cnn.test()
    cnn_result.write("\t".join([str(i) for i in cnn_res]))
    cnn.save('Fingerprint/cnn_models/{}.h5'.format(i))
    
    # plsda model
    plsda = PLSDA(spec, Y)
    plsda.train()
    plsda_res = plsda.test()
    plsda_result.write("\t".join([str(i) for i in plsda_res]))
    
    # logistic regression
    lr = LR(spec, y)
    lr.train()
    lr_res = lr.test()
    lr_result.write("\t".join([str(i) for i in lr_res]))
    
    # xgboost
    xgb = XGBoost(spec, y)
    xgb.train()
    xgb_res = xgb.test()
    xgb_result.write("\t".join([str(i) for i in xgb_res]))
    