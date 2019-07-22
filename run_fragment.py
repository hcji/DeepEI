# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 08:30:54 2019

@author: hcji
"""

import json
import numpy as np
from DeepEI.fragment import build_cnn_model, build_dnn_model

with open('Data/split.json', 'r') as js:
    keep = np.array(json.load(js)['keep'])
    
smiles = np.array(json.load(open('Data/All_smiles.json')))[keep]
morgan = np.load('Data/Morgan_fp.npy')[keep,:]
spec = np.load('Data/Peak_data.npy')[keep,:]


res_dnn = build_dnn_model(morgan, spec, save_name='dnn_model')
res_sin = build_cnn_model(smiles, spec, method='single_channel', save_name='cnn_model_single_channel')
res_mul = build_cnn_model(smiles, spec, method='multi_channel', save_name='cnn_model_multi_channel')

print ('The mean of R2 of dnn model is {}'.format(res_dnn['R2_mean']))
print ('The mean of R2 of single-channel model is {}'.format(res_sin['R2_mean']))
print ('The mean of R2 of multi-channel model is {}'.format(res_mul['R2_mean']))