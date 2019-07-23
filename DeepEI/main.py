# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:26:26 2019

@author: hcji
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


'''
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
'''

def calc_molsim(smi1, smi2):
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
    return DataStructs.DiceSimilarity(getfp(smi1), getfp(smi2))


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


def search_spec(s, ri, extand=True):
    
    
    