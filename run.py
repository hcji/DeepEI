# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:43:43 2019

@author: hcji
"""

import json
import numpy as np
from tqdm import tqdm
from DeepEI.train import build_RI_model, build_FP_models

'''
train a model to predict retention index
'''
smiles = json.load(open('Data/All_smiles.json'))
rindex = np.load('Data/RI_data.npy')
morgan = np.load('Data/Morgan_fp.npy')
cdkdes = np.load('Data/CDK_des.npy')

# remove descriptors includes nan
keep = []
for i in range(cdkdes.shape[1]):
    v = list(cdkdes[:,i])
    if np.isnan(np.min(v)):
        continue
    else:
        keep.append(i)
cdkdes = cdkdes[:, keep]

# check the number of data
n_SimiStdNP = len(np.where(~ np.isnan(rindex[:,0]))[0])
n_StdNP = len(np.where(~ np.isnan(rindex[:,1]))[0])
n_StdPolar = len(np.where(~ np.isnan(rindex[:,2]))[0])

SimiStdNP_model = build_RI_model(morgan, rindex[:,0], 'SimiStdNP')
StdPolar_model = build_RI_model(morgan, rindex[:,2], 'StdPolar')


'''
train models to predict fingerprint
'''
spec = np.load('Data/Peak_data.npy')
cdk_fp = np.load('Data/CDK_fp.npy')
output = build_FP_models(spec, cdk_fp)
output.to_csv('fingerprint_model.csv')
