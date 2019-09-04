# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:17:18 2019

@author: hcji
"""

import os
import json
import numpy as np 
from tqdm import tqdm
from tensorflow.keras.models import model_from_json, load_model
from smiles_to_onehot.encoding import one_hot_coding

def predict_fingerprint(spec):
    files = os.listdir('Model/Fingerprint')
    files = [f for f in files if '.h5' in f]
    modjs = open('Model/Fingerprint/model.json', 'r').read()
    model = model_from_json(modjs)
    pred_fp = np.zeros((spec.shape[0], len(files)))
    for i, f in enumerate(tqdm(files)):
        model.load_weights('Model/Fingerprint/' + f)
        pred = np.round(model.predict(spec))[:,0]
        pred_fp[:,i] = pred
    return pred_fp


def predict_RI(smiles, mode='SimiStdNP'):
    words = open('Model/RI/SimiStdNP_CNN_multi_dict.json', 'r').read()
    words = json.loads(words)
    if mode == 'SimiStdNP':
        model = load_model('Model/RI/SimiStdNP_CNN_multi_model.h5')
    else:
        model = load_model('Model/RI/StdPolar_CNN_multi_model.h5')
    
    X = []
    for i, smi in enumerate(smiles):
        xi = one_hot_coding(smi, words, max_len=600)
        X.append(xi.todense())
    X = np.array(X)
    pred = model.predict(X)
    return pred


def predict_MS(morgan):
    modjs = open('Model/Fragment/dnn_model.json', 'r').read()
    model = model_from_json(modjs)
    model.load_weights('Model/Fragment/dnn_model.h5')
    pred = model.predict(morgan)
    return pred
    
