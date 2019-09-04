# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:53:52 2019

@author: hcji
"""


import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from DeepEI.searchsim import get_simcomps
from DeepEI.predict import predict_fingerprint, predict_RI, predict_MS


with open('Data/split.json', 'r') as js:
    j = json.load(js)
    keep = np.array(j['keep'])
    test = np.array(j['isolate'])
    
smiles = np.array(json.load(open('Data/All_smiles.json')))[keep]
morgan = np.load('Data/Morgan_fp.npy')[keep,:]
spec = np.load('Data/Peak_data.npy')[keep,:]

test_smiles = np.array(json.load(open('Data/All_smiles.json')))[test]
test_morgan = np.load('Data/Morgan_fp.npy')[test,:]
test_spec = np.load('Data/Peak_data.npy')[test,:]
test_ri = np.load('Data/RI_data.npy')[test,:]


def calc_molsim(smi1, smi2):
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
    try:
        output = DataStructs.DiceSimilarity(getfp(smi1), getfp(smi2))
    except:
        output = 0
    return output


def get_spec_score(s, spec):
    score = []
    for ss in spec:
        score.append(pearsonr(ss, s)[0])
    return np.array(score)


def get_ri_score(ri, ris):
    dif = np.abs((ris - ri) / 1000)
    score = np.exp(-dif)
    return score
    
    
def get_fp_score(fp, fps):
    score = []
    for f in fps:
        a = len(set.intersection(*[set(f), set(fp)]))
        b = len(set.union(*[set(f), set(fp)]))
        score.append(a/float(b))
    return np.array(score)


def get_candidates(smi, thres=0.8, db='NIST'):
    if db == 'NIST':
        all_smiles = np.array(list(smiles) + list(test_smiles))
        scores = np.array([calc_molsim(smi, s) for s in all_smiles])
        candidates = all_smiles[scores > thres]
    else:
        candidates = get_simcomps(smi, thres*100)['smiles']
        refine = []
        for s in candidates:
            try:
                rf = Chem.MolToSmiles(Chem.MolFromSmiles(s))
                if '.' not in rf:
                    refine.append(rf)
            except:
                pass
        candidates = refine
    return candidates


for i, smi in enumerate(test_smiles):
    s = test_spec[i]
    r = test_ri[i,0]
    compare_scores = get_spec_score(s, spec)
    ref_smi = smiles[np.argmax(compare_scores)]
    ref_spec_score = max(compare_scores)
    candidates = get_candidates(smi, thres=ref_spec_score - 0.2, db='NIST')
    can_morgan = [np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=4096)) for s in candidates]
    can_morgan = np.array(can_morgan)
    can_pri = predict_RI(candidates)
    can_spec = predict_MS(can_morgan)
    
    