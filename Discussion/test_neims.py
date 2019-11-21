# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:29:15 2019

@author: hcji
"""

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as numpy2ri
numpy2ri.activate()
robjects.r('''source('DeepEI/rcdk.R')''')
write_sdf = robjects.globalenv['write_sdf']

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
    
    import os 
    import json
    import subprocess
    from scipy.sparse import load_npz, csr_matrix
    from libmetgem import msp
    from tqdm import tqdm
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
    from DeepEI.utils import ms2vec
    
    nist_smiles = np.array(json.load(open('Data/All_smiles.json')))
    nist_masses = np.load('Data/MolWt.npy')
    nist_spec = load_npz('Data/Peak_data.npz').todense()
    
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
    
    pred_spec = []
    output = pd.DataFrame(columns=['smiles', 'mass', 'score', 'rank'])
    for i in tqdm(range(len(smiles))):
        smi = smiles[i] # smiles
        specr = spec[i] # true spectrum
        mass = molwt[i] # true mol weight
        write_sdf(smi, 'Temp/mol.sdf')
        cwd = 'E:\\project\\deep-molecular-massspec'
        cmd = 'python make_spectra_prediction.py --input_file=E:/project/DeepEI/Temp/mol.sdf --output_file=E:/project/DeepEI/Temp/mol_anno.sdf --weights_dir=model/massspec_weights'
        subprocess.call(cmd, cwd=cwd) # predict spectrum with neims
        try:
            pred_speci = parser_NEIMS('Temp/mol_anno.sdf')
            pred_vec = ms2vec(pred_speci['mz'], pred_speci['intensity']) # spectrum to vector
            os.unlink('Temp/mol_anno.sdf')
        except:
            continue
        
        speci = spec[i]
        candidate = np.where(np.abs(nist_masses - mass) < 5)[0] # candidate of nist
        true_score = weitht_dot_product(speci, pred_vec) # score of true compound
        cand_score = get_score(speci, nist_spec[candidate,:], m='wdp') # score of nist candidates
        rank = len(np.where(cand_score > true_score)[0]) + 1 # rank
        pred_spec.append(pred_vec)
        os.unlink('Temp/mol.sdf')
        
        output.loc[len(output)] = [smi, mass, true_score, rank]
        output.to_csv('rank_neims_massbank.csv')
    pred_spec = np.array(pred_spec)
    np.save('Data/neims_spec_massbank.npy', pred_spec)
    
