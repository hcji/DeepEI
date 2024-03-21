# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:42:15 2024

@author: DELL
"""


import os
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
from tensorflow.keras.models import model_from_json, load_model

from PyFingerprint.fingerprint import get_fingerprint


mlp = pd.read_csv('Fingerprint/results/mlp_result.txt', sep='\t', header=None)
mlp.columns = ['id', 'accuracy', 'precision', 'recall', 'f1']
fpkeep = mlp['id'][np.where(mlp['f1'] > 0.5)[0]] # only keep the model with F1>0.5


def ms2vec(peakindex, peakintensity, maxmz=2000):
    output = np.zeros(maxmz)
    for i, j in enumerate(peakindex):
        if round(j) >= maxmz:
            continue
        output[int(round(j))] = float(peakintensity[i])
    output = output / (max(output) + 10 ** -6)
    return output


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
    for i, f in enumerate(files):
        model.load_weights('Fingerprint/mlp_models/' + f)
        pred = np.round(model.predict(spec))[:,0]
        pred_fp[:,i] = pred  
    return pred_fp


def get_fp_score(fp, all_fps):
    scores = np.zeros(all_fps.shape[0])
    for i in range(all_fps.shape[0]):
        fpi = all_fps[i,:]
        fpi = fpi.transpose()
        scores[i] = jaccard_score(fp, fpi)
    return scores


def get_all_fingerprints(smi):
    types=['standard', 'pubchem', 'klekota-roth', 'maccs', 'estate', 'circular']
    fp = [get_fingerprint(smi, t).to_numpy() for t in types]
    fp = list(itertools.chain(*fp))
    return fp


unknown_spectra = [[[55, 70, 145, 255], [23, 999, 344, 77]], [[58, 75, 233, 259], [23, 566, 304, 999]], [[15, 88, 170, 335], [15, 99, 999, 664]]] # not real spectra
unknown_peak_vecs = np.array([ms2vec(s[0], s[1]) for s in unknown_spectra])
pred_fps = predict_fingerprint(unknown_peak_vecs, fpkeep) 


candidate_smiles = ["CCOP(C)(=O)OP(C)(=S)OCC", "C[Si](C)(C)NC(=O)N1c2ccccc2CC(O[Si](C)(C)C)c2ccccc21", "O=C(C(Br)C(Br)c1ccccc1)C(Br)C(Br)c1ccccc1"]
candidate_fps = np.array([get_all_fingerprints(s) for s in candidate_smiles])
candidate_fps = candidate_fps[:, fpkeep] # only keep the fingerprints with the prediction model

pred_fp = pred_fps[0] # choose the first unknown compound
scores = get_fp_score(pred_fp, candidate_fps)
