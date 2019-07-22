# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:43:39 2019

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
'''

smiles = np.array(json.load(open('Data/All_smiles.json')))
morgan = np.load('Data/Morgan_fp.npy')
spec = np.load('Data/Peak_data.npy')

def search_spec(s, spec, smiles, top=50):
    score = []
    for ss in spec:
        score.append(pearsonr(s, ss)[0])
    arg = np.argsort(-np.array(score))
    tops = arg[range(top)]
    smi = smiles[tops]
    scr = np.array(score)[tops]
    return pd.DataFrame({'smiles': smi, 'scores': scr})
    

def calc_molsim(smi1, smi2):
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
    return DataStructs.DiceSimilarity(getfp(smi1), getfp(smi2))
