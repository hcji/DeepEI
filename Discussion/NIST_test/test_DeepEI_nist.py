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
from sklearn.metrics import accuracy_score, jaccard_score
from scipy.sparse import load_npz, csr_matrix
from DeepEI.predict import predict_RI, predict_fingerprint

'''
fp_data = pd.read_csv('Result/fingerprint_DNN.csv')
weights = fp_data['accuracy']
'''

def get_fp_score(fp, all_fps):
    scores = np.zeros(all_fps.shape[0])
    for i in range(all_fps.shape[0]):
        fpi = all_fps[i,:]
        fpi = fpi.transpose()
        scores[i] = jaccard_score(fp, fpi)
        # scores[i] = jaccard_score(fp, fpi, sample_weight = weights)
    return scores


if __name__ == '__main__':
    
    smiles = np.array(json.load(open('DeepEI/data/all_smiles.json')))
    masses = np.load('DeepEI/data/molwt.npy')
    with open('DeepEI/data/split.json', 'r') as js:
        split = json.load(js)
    keep = np.array(split['keep'])
    isolate = np.array(split['isolate'])
    
    # only keep fingerprint with f1 > 0.5
    mlp = pd.read_csv('Fingerprint/results/mlp_result.txt', sep='\t', header=None)
    mlp.columns = ['id', 'accuracy', 'precision', 'recall', 'f1']
    fpkeep = mlp['id'][np.where(mlp['f1'] > 0.5)[0]]
    
    # ms for test
    test_smiles = smiles[isolate]
    test_mass = masses[isolate]
    test_spec = load_npz('DeepEI/data/peakvec.npz')
    test_spec = test_spec[isolate, :].todense()
    
    # included fingerprint
    files = os.listdir('Fingerprint/mlp_models')
    rfp = np.array([int(f.split('.')[0]) for f in files if '.h5' in f])
    rfp = set(rfp).intersection(set(fpkeep))
    rfp = np.sort(list(rfp)).astype(int)
    
    cdk_fp = load_npz('DeepEI/data/fingerprints.npz')
    cdk_fp = csr_matrix(cdk_fp)[:, rfp].todense()
    
    # predict fingerprints via ms
    pred_fp = predict_fingerprint(test_spec, fpkeep)
    
    # rank
    output = pd.DataFrame(columns=['smiles', 'mass', 'fp_score', 'rank', 'candidates'])
    for i in tqdm(range(len(isolate))):
        smi = test_smiles[i]
        mass = test_mass[i]
        pred_fpi = pred_fp[i,:]
        trueindex = np.where(smiles == smi)[0][0]
        
        # mass filter
        candidate = np.where(np.abs(masses - mass) < 5)[0]
        w_true = np.where(candidate==trueindex)[0]
        if len(w_true)==0:
            rank = 99999
        else:
            fp_scores = get_fp_score(pred_fpi, cdk_fp[candidate, :])
            true_fp_score = fp_scores[w_true[0]]
            rank = len(np.where(fp_scores > true_fp_score)[0]) + 1 
        output.loc[len(output)] = [smi, mass, true_fp_score, rank, len(candidate)]
    output.to_csv('Discussion/NIST_test/results/DeepEI_nist.csv')
        