# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:34:57 2020

@author: hcji
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:29:15 2019

@author: hcji
"""

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import load_npz
from DeepEI.utils import get_score, get_fp_score
from DeepEI.predict import predict_fingerprint

with open('DeepEI/data/split.json', 'r') as js:
    split = json.load(js)
keep = np.array(split['keep'])

nist_smiles = np.array(json.load(open('DeepEI/data/all_smiles.json')))[keep]
nist_masses = np.load('DeepEI/data/molwt.npy')[keep]
nist_spec = load_npz('DeepEI/data/peakvec.npz').todense()[keep,:]
nist_fingerprint = load_npz('DeepEI/data/fingerprints.npz').todense()[keep,:]
neims_nist_spec = load_npz('DeepEI/data/neims_spec_nist.npz').todense()[keep,:]

neims_msbk_smiles = np.array(json.load(open('DeepEI/data/neims_msbk_smiles.json')))
neims_msbk_masses = np.load('DeepEI/data/neims_msbk_masses.npy')
neims_msbk_spec = load_npz('DeepEI/data/neims_spec_msbk.npz').todense()
neims_msbk_cdkfps = load_npz('DeepEI/data/neims_msbk_cdkfps.npz').todense()

msbk_smiles = np.array(json.load(open('DeepEI/data/msbk_smiles.json')))
msbk_masses = np.load('DeepEI/data/msbk_masses.npy')
msbk_spec = load_npz('DeepEI/data/msbk_spec.npz').todense()

mlp = pd.read_csv('Fingerprint/results/mlp_result.txt', sep='\t', header=None)
mlp.columns = ['id', 'accuracy', 'precision', 'recall', 'f1']
fpkeep = mlp['id'][np.where(mlp['f1'] > 0.5)[0]]
pred_fps = predict_fingerprint(msbk_spec, fpkeep) 

db_smiles = np.array(list(nist_smiles) + list(neims_msbk_smiles))
db_masses = np.append(nist_masses, neims_msbk_masses)
db_spec = np.append(neims_nist_spec, neims_msbk_spec, axis=0)
db_fingerprints = np.append(nist_fingerprint, neims_msbk_cdkfps, axis=0)[:, fpkeep]


if __name__ == '__main__':
    
    output = pd.DataFrame(columns=['smiles', 'mass', 'score', 'rank', 'inNIST'])
    for i, smi in enumerate(tqdm(msbk_smiles)):

        specr = msbk_spec[i] # true spectrum
        mass = msbk_masses[i] # true mol weight
        pred_fp = pred_fps[i]
        incl = smi in nist_smiles
        
        candidate = np.where(np.abs(db_masses - mass) < 5)[0]
        cand_smi = db_smiles[candidate]
        try:
            wh_true = np.where(cand_smi == smi)[0][0]
        except:
            continue
        scores_a = np.array(get_score(specr, db_spec[candidate,:], m='wdp'))
        scores_b = get_fp_score(pred_fp, db_fingerprints[candidate, :])
        
        scores = 0.3*scores_a + 0.7*scores_b     
        true_score = scores[wh_true]
        rank = len(np.where(scores > true_score)[0]) + 1
        
        output.loc[len(output)] = [smi, mass, true_score, rank, incl]
        output.to_csv('Discussion/results/conbine_massbank.csv')