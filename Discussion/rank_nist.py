# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:53:47 2019

@author: hcji
"""

# rank in NIST
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.sparse import load_npz
from DeepEI.predict import predict_RI, predict_fingerprint


def get_fp_score(fp, all_fps):
    scores = np.zeros(all_fps.shape[0])
    for i in range(all_fps.shape[0]):
        fpi = all_fps[i,:]
        fpi = fpi.transpose()
        scores[i] = accuracy_score(fp, fpi)
    return scores


if __name__ == '__main__':
    
    smiles = np.array(json.load(open('Data/All_smiles.json')))
    masses = np.load('Data/MolWt.npy')
    with open('Data/split.json', 'r') as js:
        split = json.load(js)
    keep = np.array(split['keep'])
    isolate = np.array(split['isolate'])
    
    # ms for test
    test_smiles = smiles[isolate]
    test_mass = masses[isolate]
    test_spec = load_npz('Data/Peak_data.npz')
    test_spec = test_spec[isolate, :].todense()
    test_ri = np.load('Data/RI_data.npy')[isolate,0]
    
    # included fingerprint
    files = os.listdir('Model/Fingerprint')
    rfp = np.array([int(f.split('.')[0]) for f in files if '.h5' in f])
    rfp = np.sort(rfp)
    
    rindex = predict_RI(smiles, mode='SimiStdNP')[:,0] # predicted ri
    cdk_fp = load_npz('Data/CDK_fp.npz')
    cdk_fp = cdk_fp[:, rfp].todense()
    
    # predict fingerprints via ms
    pred_fp = predict_fingerprint(test_spec)
    
    # rank
    output = pd.DataFrame(columns=['smiles', 'mass', 'fp_score', 'rank_1', 'rank_2', 'rank_3'])
    for i in tqdm(range(len(isolate))):
        smi = test_smiles[i]
        mass = test_mass[i]
        ri = test_ri[i]
        pred_fpi = pred_fp[i,:]
        trueindex = np.where(smiles == smi)[0][0]
        
        # with ri filter
        candidate_1 = np.where(np.abs(rindex - ri) < 200)[0] # ri filter with 200
        w_true = np.where(candidate_1==trueindex)[0]
        if len(w_true)==0:
            rank_1 = 9999
        else:
            fp_scores = get_fp_score(pred_fpi, cdk_fp[candidate_1, :])
            true_fp_score = fp_scores[w_true[0]]
            rank_1 = len(np.where(fp_scores > true_fp_score)[0]) + 1
            
        # with mass filter
        candidate_2 = np.where(np.abs(masses - mass) < 10)[0]
        w_true = np.where(candidate_2==trueindex)[0]
        if len(w_true)==0:
            rank_2 = 9999
        else:
            fp_scores = get_fp_score(pred_fpi, cdk_fp[candidate_2, :])
            true_fp_score = fp_scores[w_true[0]]
            rank_2 = len(np.where(fp_scores > true_fp_score)[0]) + 1        
        
        # with ri and mass filter
        candidate_3 = np.where(np.abs(masses - mass) < 10)[0]
        w_true = np.where(candidate_3==trueindex)[0]
        if len(w_true)==0:
            rank_3 = 9999
        else:
            fp_scores = get_fp_score(pred_fpi, cdk_fp[candidate_3, :])
            true_fp_score = fp_scores[w_true[0]]
            rank_3 = len(np.where(fp_scores > true_fp_score)[0]) + 1        
        
        output.loc[len(output)] = [smi, mass, true_fp_score, rank_1, rank_2, rank_3]
        output.to_csv('rank_nist.csv')
        