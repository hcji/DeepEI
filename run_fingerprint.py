# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:43:43 2019

@author: hcji
"""

import json
import numpy as np
from DeepEI.fingerprint import build_FP_models

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
train models to predict fingerprint
'''
spec = np.load('Data/Peak_data.npy')[keep,:]
cdk_fp = np.load('Data/CDK_fp.npy')[keep,:]
output = build_FP_models(spec, cdk_fp, method='PLSDA', check=False)
output.to_csv('fingerprint_model_dnn.csv')

