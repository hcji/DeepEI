# -*- coding: utf-8 -*-
"""
Created on Sat May  2 09:31:18 2020

@author: zmzhang
"""



import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import load_npz, csr_matrix

from Fingerprint.mlp import MLP


# load data
with open('DeepEI/data/split.json', 'r') as js:
    keep = np.array(json.load(js)['keep'])
spec = load_npz('DeepEI/data/peakvec.npz')
fps = load_npz('DeepEI/data/fingerprints.npz')

# exclude isolate test data
spec = spec.todense()[keep,:]
fps = csr_matrix(fps)[keep,:]

# build model
for i in tqdm(range(fps.shape[1])):
    y = fps[:,i].todense()
    y = np.squeeze(np.asarray(y))
    
    # permutation
    y = np.random.permutation(y)
    
    # check: 0.1 < bias < 0.9
    fr = np.sum(y) / len(y)
    if (fr < 0.1) or (fr > 0.9):
        continue
    Y = np.vstack((y, (1-y))).transpose()
    
    mlp_perm_result = open('Fingerprint/results/mlp_perm_result.txt', 'a+')
    
    # mlp model
    mlp = MLP(spec, Y)
    mlp.train()
    mlp_res = mlp.test()
    mlp_perm_result.write("\t".join([str(i)] + [str(j) for j in mlp_res]))
    mlp_perm_result.write("\n")

    mlp_perm_result.close()
