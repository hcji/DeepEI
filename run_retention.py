# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:21:17 2019

@author: hcji
"""


import json
import numpy as np
from DeepEI.read import collect
from DeepEI.retention import build_RI_model_descriptor, build_RI_model_CNN, build_RI_model_RNN

smiles = json.load(open('Data/All_smiles.json'))
'''
isolate = list(np.random.choice(range(len(smiles)), 1000))
keep = [i for i in range(len(smiles)) if i not in isolate]
isolate = [int(i) for i in isolate]
keep = [int(i) for i in keep]
split = {'isolate': isolate, 'keep': keep}
with open('Data/split.json', 'w') as js:
    json.dump(split, js)
'''

with open('Data/split.json', 'r') as js:
    keep = np.array(json.load(js)['keep'])


'''
train a model to predict retention index
'''
smiles = np.array(json.load(open('Data/All_smiles.json')))[keep]
rindex = np.load('Data/RI_data.npy')[keep,:]

morgan = np.load('Data/Morgan_fp.npy')[keep,:]
cdkdes = np.load('Data/CDK_des.npy')[keep,:]

# remove descriptors includes nan
keep1 = []
for i in range(cdkdes.shape[1]):
    v = list(cdkdes[:,i])
    if np.isnan(np.min(v)):
        continue
    else:
        keep1.append(i)
cdkdes = cdkdes[:, keep1]


# check the number of data
n_SimiStdNP = len(np.where(~ np.isnan(rindex[:,0]))[0])
n_StdNP = len(np.where(~ np.isnan(rindex[:,1]))[0])
n_StdPolar = len(np.where(~ np.isnan(rindex[:,2]))[0])

SimiStdNP_model_dnn = build_RI_model_descriptor(morgan, cdkdes, rindex[:,0], 'morgan', 'SimiStdNP_DNN')
StdPolar_model_dnn = build_RI_model_descriptor(morgan, cdkdes, rindex[:,2], 'morgan', 'StdPolar_DNN')

SimiStdNP_model_cnn = build_RI_model_CNN(smiles, rindex[:,0], 'multi_channel', 'SimiStdNP_CNN')
StdPolar_model_cnn = build_RI_model_CNN(smiles, rindex[:,2], 'multi_channel', 'StdPolar_CNN')
