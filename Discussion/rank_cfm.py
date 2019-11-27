# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 08:39:09 2019

@author: hcji
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import load_npz, csr_matrix
from DeepEI.utils import ms2vec, vec2ms
from PyCFMID.PyCFMID import cfm_predict


smiles = np.array(json.load(open('Data/All_smiles.json')))
masses = np.load('Data/MolWt.npy')
with open('Data/split.json', 'r') as js:
    split = json.load(js)
keep = np.array(split['keep'])
isolate = np.array(split['isolate'])
    
test_smiles = smiles[isolate]
test_mass = masses[isolate]
test_spec = load_npz('Data/Peak_data.npz')
test_spec = test_spec[isolate, :].todense()
    
    
def gen_cfm_spec(i):
    smi = test_smiles[i]
    try:
        res = cfm_predict(smi, ion_source='EI')
        spec = ms2vec(res['low_energy']['mz'], res['low_energy']['intensity'])
    except:
        spec = np.zeros(2000)
    return spec

def dot_product(a, b):
    a = np.squeeze(np.asarray(a))
    b = np.squeeze(np.asarray(b))
    return np.dot(a,b)/ np.sqrt((np.dot(a,a)* np.dot(b,b)))

def weitht_dot_product(a, b):
    a = np.squeeze(np.asarray(a))
    b = np.squeeze(np.asarray(b))
    w = np.arange(len(a))
    wa = np.sqrt(a) * w
    wb = np.sqrt(b) * w
    return np.dot(wa,wb) / np.sqrt((np.dot(wa,wa)* np.dot(wb,wb)))

def get_score(x, X, m='dp'):
    if m == 'dp':
        s = [dot_product(x, X[i,:]) for i in range(X.shape[0])]
    else:
        s = [weitht_dot_product(x, X[i,:]) for i in range(X.shape[0])]
    return s
    


if __name__ == '__main__':
    '''
    from joblib import Parallel, delayed
    
    cfm_spec = Parallel(n_jobs=8, verbose=5)(delayed(gen_cfm_spec)(i) for i in range(len(test_smiles)))
    cfm_spec = np.array(cfm_spec)
    np.save('Data/cfm_spec.npy', cfm_spec)
    '''
    
    cfm_spec = np.load('Data/cfm_spec.npy')
    exp_spec = load_npz('Data/Peak_data.npz')
    exp_spec = exp_spec[keep, :].todense()
    aug_spec = np.vstack([exp_spec, cfm_spec])
    aug_smi = np.concatenate([smiles[keep], smiles[isolate]])
    aug_mass = np.concatenate([masses[keep], masses[isolate]])
    
    output = pd.DataFrame(columns=['smiles', 'mass', 'dp_score', 'rank_5da', 'rank_1da'])
    for i in tqdm(range(len(isolate))):
        smi = test_smiles[i]
        mass = test_mass[i]
        spec = test_spec[i]
        trueindex = np.where(aug_smi == smi)[0][0]
        '''
        mz, intensity = vec2ms(np.squeeze(np.asarray(spec)))
        mza, intensitya = vec2ms(np.squeeze(np.asarray(aug_spec[trueindex])))
        plt.vlines(mz, np.zeros(len(mz)), intensity, color='red')
        plt.vlines(mza, np.zeros(len(mza)), -intensitya, color='blue')
        '''
        candidate_1 = np.where(np.abs(aug_mass - mass) < 5)[0]
        w_true_1 = np.where(candidate_1==trueindex)[0][0]
        scores_1 = get_score(spec, aug_spec[candidate_1,:], m='dp')
        
        candidate_2 = np.where(np.abs(aug_mass - mass) < 1)[0]
        w_true_2 = np.where(candidate_2==trueindex)[0][0]
        scores_2 = get_score(spec, aug_spec[candidate_2,:], m='dp')
        
        true_dp_score_1 = scores_1[w_true_1]
        true_dp_score_2 = scores_2[w_true_2]
        
        rank_1 = len(np.where(scores_1 > true_dp_score_1)[0]) + 1
        rank_2 = len(np.where(scores_2 > true_dp_score_2)[0]) + 1
        
        output.loc[len(output)] = [smi, mass, true_dp_score_1, rank_1, rank_2]
        output.to_csv('rank_cfm.csv')        