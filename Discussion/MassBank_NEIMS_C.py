# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:25:33 2019

@author: hcji
"""

import json
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from libmetgem import msp
from tqdm import tqdm
from rdkit import Chem
from DeepEI.utils import get_score
from DeepEI.predict import predict_fingerprint

with open('DeepEI/data/split.json', 'r') as js:
    split = json.load(js)
keep = np.array(split['keep'])

nist_smiles = np.array(json.load(open('DeepEI/data/all_smiles.json')))[keep]
nist_masses = np.load('DeepEI/data/molwt.npy')[keep]
nist_spec = load_npz('DeepEI/data/peakvec.npz').todense()[keep,:]
neims_nist_spec = load_npz('DeepEI/data/neims_spec_nist.npz').todense()[keep,:]

neims_msbk_smiles = np.array(json.load(open('DeepEI/data/neims_msbk_smiles.json')))
neims_msbk_masses = np.load('DeepEI/data/neims_msbk_masses.npy')
neims_msbk_spec = load_npz('DeepEI/data/neims_spec_msbk.npz').todense()

msbk_smiles = np.array(json.load(open('DeepEI/data/msbk_smiles.json')))
msbk_masses = np.load('DeepEI/data/msbk_masses.npy')
msbk_spec = load_npz('DeepEI/data/msbk_spec.npz').todense()

db_smiles = np.array(list(nist_smiles) + list(neims_msbk_smiles))
db_masses = np.append(nist_masses, neims_msbk_masses)
db_spec_a = np.append(nist_spec, neims_msbk_spec, axis=0)
db_spec_b = np.append(neims_nist_spec, neims_msbk_spec, axis=0)


if __name__ == '__main__':
    
    output = pd.DataFrame(columns=['smiles', 'mass', 'score', 'rank', 'candidates', 'inNIST'])
    for i, smi in enumerate(tqdm(msbk_smiles)):
        mass = msbk_masses[i]
        specr = msbk_spec[i]
        incl = smi in nist_smiles
        
        candidate = np.where(np.abs(nist_masses - mass) < 5)[0]
        cand_smi = nist_smiles[candidate]
        scores = get_score(specr, nist_spec[candidate,:], m='wdp')
        
        if max(scores) > 0.95:
            try:
                wh_true = np.where(cand_smi == smi)[0][0]
                true_score = scores[wh_true]
                rank = len(np.where(scores > true_score)[0]) + 1
            except:
                true_score = 0
                rank = 99999
            output.loc[len(output)] = [smi, mass, true_score, rank, len(candidate), incl]
            output.to_csv('Discussion/results/NEIMS_massbank_C.csv')
            continue
        else:
            candidate = np.where(np.abs(db_masses - mass) < 5)[0] # candidates  
            cand_smi = db_smiles[candidate]
            try:
                wh_true = np.where(cand_smi == smi)[0][0]
            except:
                continue
            scores = get_score(specr, db_spec_b[candidate, :]) # scores of all candidtates
            true_score = scores[wh_true]
            rank = len(np.where(scores > true_score)[0]) + 1
            output.loc[len(output)] = [smi, mass, true_score, rank, len(candidate), incl]
            output.to_csv('Discussion/results/NEIMS_massbank_C.csv')