# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:00:47 2019

@author: hcji
"""

import os
import json
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from DeepEI.utils import ms2vec
from rdkit import Chem
from io import StringIO

def writeSDF(smi, file):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError('invalid smiles')
    sio = StringIO()
    w = Chem.SDWriter(sio)
    w.write(m)
    w=None
    string = sio.getvalue()
    with open(file, 'w') as f:
        f.write(string)
    

with open('DeepEI/data/split.json', 'r') as js:
    j = json.load(js)
    keep = np.array(j['keep'])
    test = np.array(j['isolate'])

smiles = np.array(json.load(open('DeepEI/data/all_smiles.json')))

def parser_NEIMS(sdf):
    with open(sdf) as t:
        cont = t.readlines()
    start = 0
    spec = pd.DataFrame(columns=['mz', 'intensity'])
    for l in cont:
        l = l.replace('\n', '')
        if '$$$$' in l:
            break
        if start == 1:
            if l != '':
                l = l.split(' ')
                spec.loc[len(spec)] = np.array(l).astype(int)
        if 'SPECTRUM' in l:
            start = 1
    return spec

def dot_product(a, b):
    a = np.squeeze(np.asarray(a))
    b = np.squeeze(np.asarray(b))
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def weitht_dot_product(a, b):
    a = np.squeeze(np.asarray(a))
    b = np.squeeze(np.asarray(b))
    w = np.arange(len(a)) + 1
    wa = np.sqrt(a) * w
    wb = np.sqrt(b) * w
    return np.dot(wa,wb) / (np.linalg.norm(wa) * np.linalg.norm(wb))

def get_score(x, X, m='wdp'):
    if m == 'dp':
        s = [dot_product(x, X[i,:]) for i in range(X.shape[0])]
    else:
        s = [weitht_dot_product(x, X[i,:]) for i in range(X.shape[0])]
    return s

# predict ms of the isolate compounds
# run once, then save
pred_spec = np.zeros((len(test), 2000))
for a, i in enumerate(tqdm(test)):
    smi = smiles[i]
    writeSDF(smi, 'Temp/mol.sdf')
    cwd = 'E:\\project\\deep-molecular-massspec'
    cmd = 'python make_spectra_prediction.py --input_file=E:/project/DeepEI/Temp/mol.sdf --output_file=E:/project/DeepEI/Temp/mol_anno.sdf --weights_dir=model/massspec_weights'
    subprocess.call(cmd, cwd=cwd)
    try:
        speci = parser_NEIMS('Temp/mol_anno.sdf')
        pred_vec = ms2vec(speci['mz'], speci['intensity'])
        os.unlink('Temp/mol_anno.sdf')
    except:
        pred_vec = np.zeros(2000) #  # if error, use a zero vec as placeholder. but it won't count when comparsion.
    pred_spec[a,:] = pred_vec
    os.unlink('Temp/mol.sdf')
np.save('Discussion/NIST_test/neims_spec_nist.npy', pred_spec)

if __name__ == '__main__':
    
    from scipy.sparse import load_npz
    
    masses = np.load('DeepEI/data/molwt.npy')
    spec = load_npz('DeepEI/data/peakvec.npz').todense()

    # ms for test
    test_smiles = smiles[test]
    test_mass = masses[test]
    test_spec = spec[test,:]
    
    pred_spec = np.load('Discussion/NIST_test/neims_spec_nist.npy')
    
    output = pd.DataFrame(columns=['smiles', 'mass', 'score', 'rank'])
    for i in tqdm(range(len(test))):
        smi = test_smiles[i]
        mass = test_mass[i]
        true_vec = test_spec[i]
        pred_vec = pred_spec[i]
        
        if np.sum(pred_vec) == 0:
            continue
        
        true_score = weitht_dot_product(true_vec, pred_vec) # score of true candidate

        candidate = np.where(np.abs(masses - mass) < 5)[0] # candidate of nist
        cand_smi = smiles[candidate]
        rep_ind = np.where(cand_smi == smi)[0] # if the compound in nist, remove it.
        candidate = np.delete(candidate, rep_ind)
        cand_score = get_score(true_vec, spec[candidate,:], m='wdp') # score of nist candidates
        
        rank = len(np.where(cand_score > true_score)[0]) + 1 # rank
        output.loc[len(output)] = [smi, mass, true_score, rank]
    output.to_csv('Discussion/NIST_test/results/neims_nist.csv')
            