# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:15:36 2019

@author: hcji
"""

import os
import json
import numpy as np 
from tqdm import tqdm
from tensorflow.keras.models import model_from_json, load_model
from smiles_to_onehot.encoding import one_hot_coding

def predict_RI(smiles, mode='SimiStdNP'):
    words = open('DeepEI/data/words.json', 'r').read()
    words = json.loads(words)
    if mode == 'SimiStdNP':
        model = load_model('Retention/models/SimiStdNP_CNN_multi_model.h5')
    elif mode == 'StdNP':
        model = load_model('Retention/models/StdNP_CNN_multi_model.h5')
    elif mode == 'StdPolar':
        model = load_model('Retention/models/StdPolar_CNN_multi_model.h5')
    else:
        return None
    
    X = []
    for i, smi in enumerate(smiles):
        xi = one_hot_coding(smi, words, max_len=100)
        X.append(xi.todense())
    X = np.array(X)
    pred = model.predict(X)
    return pred


def predict_fingerprint(spec, fpkeep):
    files = os.listdir('Fingerprint/mlp_models')
    rfp = np.array([int(f.split('.')[0]) for f in files if '.h5' in f])
    rfp = np.sort(rfp)
    
    rfp = set(rfp).intersection(set(fpkeep))
    rfp = np.sort(list(rfp)).astype(int)
    
    files = [str(f) + '.h5' for f in rfp]
    modjs = open('Fingerprint/mlp_models/model.json', 'r').read()
    model = model_from_json(modjs)
    pred_fp = np.zeros((spec.shape[0], len(files)))
    for i, f in enumerate(tqdm(files)):
        model.load_weights('Fingerprint/mlp_models/' + f)
        pred = np.round(model.predict(spec))[:,0]
        pred_fp[:,i] = pred  
    return pred_fp