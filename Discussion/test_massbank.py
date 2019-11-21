# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:25:33 2019

@author: hcji
"""

import numpy as np 
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics import accuracy_score, jaccard_score
from libmetgem import msp
from DeepEI.utils import ms2vec, vec2ms, get_cdk_fingerprints

def get_fp_score(fp, all_fps):
    scores = np.zeros(all_fps.shape[0])
    for i in range(all_fps.shape[0]):
        fpi = all_fps[i,:]
        fpi = fpi.transpose()
        scores[i] = jaccard_score(fp, fpi)
        # scores[i] = jaccard_score(fp, fpi, sample_weight = weights)
    return scores

if __name__ == '__main__':
    
    import os
    import json
    import pandas as pd
    from tqdm import tqdm
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
    from DeepEI.predict import predict_fingerprint
    
    data = msp.read('Data/GCMS DB_AllPublic-KovatsRI-VS2.msp')
    smiles = []
    spec = []
    molwt = []
    for i, (param, ms) in enumerate(tqdm(data)):
        smi = param['smiles']
        try:
            mass = CalcExactMolWt(Chem.MolFromSmiles(smi))
        except:
            continue
        molwt.append(mass)
        smiles.append(smi)
        spec.append(ms2vec(ms[:,0], ms[:,1]))
    
    spec = np.array(spec)
    pred_fps = predict_fingerprint(spec) # predict fingerprint of the "unknown"
    
    files = os.listdir('Model/Fingerprint')
    rfp = np.array([int(f.split('.')[0]) for f in files if '.h5' in f])
    rfp = np.sort(rfp) # the index of the used fingerprint
    
    nist_smiles = np.array(json.load(open('Data/All_smiles.json')))
    nist_masses = np.load('Data/MolWt.npy')
    nist_fps = load_npz('Data/CDK_fp.npz')
    nist_fps = csr_matrix(nist_fps)[:, rfp].todense() # fingerprints of nist compounds
    
    output = pd.DataFrame(columns=['smiles', 'mass', 'fp_score', 'rank'])
    for i in tqdm(range(len(smiles))):
        smi = smiles[i]
        mass = molwt[i]
        pred_fp = pred_fps[i]
        try:
            true_fp = np.array(get_cdk_fingerprints(smi)) # true fingerprint of the "unknown"
        except:
            continue
        true_fp = true_fp[rfp]
        true_score = jaccard_score(pred_fp, true_fp)  # score of the true compound
        
        candidate = np.where(np.abs(nist_masses - mass) < 5)[0]
        fp_scores = get_fp_score(pred_fp, nist_fps[candidate, :]) # scores of all candidtates
        rank = len(np.where(fp_scores > true_score)[0]) + 1
        
        output.loc[len(output)] = [smi, mass, true_score, rank]
        output.to_csv('rank_massbank.csv')
        