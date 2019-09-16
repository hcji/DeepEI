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

with open('Data/split.json', 'r') as js:
    keep = np.array(json.load(js)['keep'])


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

SimiStdNP_model_dnn_morgan = build_RI_model_descriptor(morgan, cdkdes, rindex[:,0], 'morgan', 'SimiStdNP_DNN_morgan')
StdPolar_model_dnn_morgan = build_RI_model_descriptor(morgan, cdkdes, rindex[:,2], 'morgan', 'StdPolar_DNN_morgan')
SimiStdNP_model_dnn_desc = build_RI_model_descriptor(morgan, cdkdes, rindex[:,0], 'descriptor', 'SimiStdNP_DNN_descriptor')
StdPolar_model_dnn_desc = build_RI_model_descriptor(morgan, cdkdes, rindex[:,2], 'descriptor', 'StdPolar_DNN_descriptor')
SimiStdNP_model_dnn_all = build_RI_model_descriptor(morgan, cdkdes, rindex[:,0], 'all', 'SimiStdNP_DNN_all')
StdPolar_model_dnn_all = build_RI_model_descriptor(morgan, cdkdes, rindex[:,2], 'all', 'StdPolar_DNN_all')
SimiStdNP_model_cnn_sin = build_RI_model_CNN(smiles, rindex[:,0], 'single_channel', 'SimiStdNP_CNN_single')
StdPolar_model_cnn_sin = build_RI_model_CNN(smiles, rindex[:,2], 'single_channel', 'StdPolar_CNN_single')
SimiStdNP_model_cnn_mul = build_RI_model_CNN(smiles, rindex[:,0], 'multi_channel', 'SimiStdNP_CNN_multi')
StdPolar_model_cnn_mul = build_RI_model_CNN(smiles, rindex[:,2], 'multi_channel', 'StdPolar_CNN_multi')
