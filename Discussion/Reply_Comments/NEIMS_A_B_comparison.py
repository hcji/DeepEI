# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:04:52 2020

@author: hcji
"""


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import load_npz
from DeepEI.utils import get_score

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

i = 70
smi = msbk_smiles[i]
specr = msbk_spec[i]
mass = msbk_masses[i]
candidate = np.where(np.abs(db_masses - mass) < 5)[0]
cand_smi = db_smiles[candidate]

scores_a = get_score(specr, db_spec_a[candidate,:], m='wdp')
scores_b = get_score(specr, db_spec_b[candidate,:], m='wdp')

wh_true = np.where(cand_smi == smi)[0][0]
true_score_a = scores_a[wh_true]
true_score_b = scores_b[wh_true]
rank_a = len(np.where(scores_a > true_score_a)[0]) + 1
rank_b = len(np.where(scores_b > true_score_b)[0]) + 1

true = candidate[wh_true]
j = candidate[435]

decoy_smi = db_smiles[j]

plt.figure(figsize=(6, 6))
plt.vlines(np.arange(0, 2000), np.zeros(2000), db_spec_a[j], 'red', label='NIST_decoy')
plt.vlines(np.arange(0, 2000), np.zeros(2000), -db_spec_b[j], 'blue', label='NEIMS_decoy')
plt.axhline(0, color='black')
plt.xlim(0, 500)
plt.legend()

plt.figure(figsize=(6, 6))
plt.vlines(np.arange(0, 2000), np.zeros(2000), specr, 'green', label='MassBank_true')
plt.axhline(0, color='black')
plt.vlines(np.arange(0, 2000), np.zeros(2000), -db_spec_b[j], 'blue', label='NEIMS_decoy')
plt.xlim(0, 500)
plt.legend()

plt.figure(figsize=(6, 6))
plt.vlines(np.arange(0, 2000), np.zeros(2000), specr, 'green', label='MassBank_true')
plt.axhline(0, color='black')
plt.vlines(np.arange(0, 2000), np.zeros(2000), -db_spec_a[j], 'red', label='NIST_decoy')
plt.xlim(0, 500)
plt.legend()

plt.figure(figsize=(6, 6))
plt.vlines(np.arange(0, 2000), np.zeros(2000), specr, 'green', label='MassBank_true')
plt.axhline(0, color='black')
plt.vlines(np.arange(0, 2000), np.zeros(2000), -db_spec_a[true], 'purple', label='NEIMS_true')
plt.xlim(0, 500)
plt.legend()
