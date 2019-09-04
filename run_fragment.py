# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 08:30:54 2019

@author: hcji
"""

import json
import numpy as np
from tensorflow.keras.models import model_from_json
from DeepEI.fragment import build_cnn_model, build_dnn_model, refine_spec

with open('Data/split.json', 'r') as js:
    keep = np.array(json.load(js)['keep'])
    
smiles = np.array(json.load(open('Data/All_smiles.json')))[keep]
morgan = np.load('Data/Morgan_fp.npy')[keep,:]
spec = np.load('Data/Peak_data.npy')[keep,:]


# res_dnn = build_dnn_model(morgan, spec, save_name='dnn_model')
res_sin = build_cnn_model(smiles, spec, method='single_channel', save_name='cnn_model_single_channel')
res_mul = build_cnn_model(smiles, spec, method='multi_channel', save_name='cnn_model_multi_channel')

# print ('The mean of R2 of dnn model is {}'.format(res_dnn['R2_mean']))
print ('The mean of R2 of single-channel model is {}'.format(res_sin['R2_mean']))
print ('The mean of R2 of multi-channel model is {}'.format(res_mul['R2_mean']))

'''
# generate simulated spectra
smiles = np.array(json.load(open('Data/All_smiles.json')))
morgan = np.load('Data/Morgan_fp.npy')

with open('Model/Fragment/dnn_model.json') as js:
    model = model_from_json(js.read())
model.load_weights('Model/Fragment/dnn_model_forward.h5')

simu_spec = model.predict(morgan)
simu_spec_refi = np.array([refine_spec(s) for s in simu_spec])
np.save('Data/Simu_Spec.npy', np.array(simu_spec_refi))
'''