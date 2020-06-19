# -*- coding: utf-8 -*-
"""
Created on Tue May  5 08:43:28 2020

@author: hcji
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import load_npz, csr_matrix, save_npz
from tqdm import tqdm
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

with open('DeepEI/data/split.json', 'r') as js:
    split = json.load(js)
keep = np.array(split['keep'])

nist_smiles = np.array(json.load(open('DeepEI/data/all_smiles.json')))[keep]
nist_fp = []
for i in tqdm(range(len(nist_smiles))):
    m = Chem.MolFromSmiles(nist_smiles[i])
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=4096)
    nist_fp.append(fp)

msbk_res = pd.read_csv('Discussion/results/DeepEI_massbank.csv')
msbk_score = msbk_res['fp_score']
msbk_nsim = []
for i in tqdm(range(len(msbk_res))):
    m = Chem.MolFromSmiles(msbk_res['smiles'][i])
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=4096)
    sims = np.array([DataStructs.FingerprintSimilarity(fp, n, metric=DataStructs.DiceSimilarity) for n in nist_fp])
    nsim = len(np.where(sims > 0.8)[0])
    msbk_nsim.append(nsim)
wh = np.where(msbk_res['inNIST'])[0]
msbk_nsim = np.array(msbk_nsim)

plt.scatter(msbk_nsim[wh], msbk_score[wh], marker='o', alpha=0.7)
plt.xlabel('Number of similar compounds')
plt.ylabel('FP score')
plt.xlim(-20, 500)

hmdb_smiles = json.load(open('DeepEI/data/hmdb_smiles.json'))
chebi_smiles = json.load(open('DeepEI/data/chebi_smiles.json'))

hmdb_fp = []
chebi_fp = []

hmdb_smiles_new = []
chebi_smiles_new = []

for smi in tqdm(hmdb_smiles):
    try:
        mol = Chem.MolFromSmiles(smi)
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 4096))
    except:
        continue
    hmdb_smiles_new.append(smi)
    hmdb_fp.append(fp)
        
for smi in tqdm(chebi_smiles):
    try:
        mol = Chem.MolFromSmiles(smi)
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 4096))
    except:
        continue
    chebi_smiles_new.append(smi)
    chebi_fp.append(fp)

chebi_fp = np.array(chebi_fp)
hmdb_fp = np.array(hmdb_fp)

nist_fp = csr_matrix(nist_fp)
save_npz('nist_fp.npz', nist_fp)

chebi_fp = csr_matrix(chebi_fp)
save_npz('chebi_fp.npz', chebi_fp)

hmdb_fp = csr_matrix(hmdb_fp)
save_npz('hmdb_fp.npz', hmdb_fp)


hmdb_fp = load_npz('hmdb_fp.npz')
nist_fp = load_npz('nist_fp.npz')
chebi_fp = load_npz('chebi_fp.npz')
chebi_fp = chebi_fp.todense()
nist_fp = nist_fp.todense()
hmdb_fp = hmdb_fp.todense()


X_embedded = np.load('X_embedded.npy')
X1_embedded = np.load('X1_embedded.npy')
y = np.append(np.zeros(len(nist_fp)), np.ones(len(chebi_fp)))
y1 = np.append(np.zeros(len(nist_fp)), np.ones(len(hmdb_fp)))

# PCA analysis for HMDB
X = np.append(nist_fp, hmdb_fp, axis=0)
y = np.append(np.zeros(len(nist_fp)), np.ones(len(hmdb_fp)))
target_names = ['NIST', 'HMDB']

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

# plt.figure(figsize=(8,6))
colors = ['blue', 'red']

for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.5, marker='.',
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel('PC1 ({} %)'.format(round( pca.explained_variance_ratio_[0]*100, 2) ))
plt.ylabel('PC2 ({} %)'.format(round( pca.explained_variance_ratio_[1]*100, 2) ))
plt.show()

# PCA analysis for ChEBI
X = np.append(nist_fp, chebi_fp, axis=0)
y = np.append(np.zeros(len(nist_fp)), np.ones(len(chebi_fp)))
target_names = ['NIST', 'ChEBI']

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

# plt.figure(figsize=(8,6))
colors = ['blue', 'green']

for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.5, marker='.',
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel('PC1 ({} %)'.format(round( pca.explained_variance_ratio_[0]*100, 2) ))
plt.ylabel('PC2 ({} %)'.format(round( pca.explained_variance_ratio_[1]*100, 2) ))
plt.show()


