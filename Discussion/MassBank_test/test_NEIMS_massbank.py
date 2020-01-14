# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:29:15 2019

@author: hcji
"""

import numpy as np
import pandas as pd
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
    w = np.arange(len(a))
    wa = np.sqrt(a) * w
    wb = np.sqrt(b) * w
    return np.dot(wa,wb) / (np.linalg.norm(wa) * np.linalg.norm(wb))

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
    import matplotlib.pyplot as plt
    from scipy.sparse import load_npz, csr_matrix
    from libmetgem import msp
    from tqdm import tqdm
    from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
    from DeepEI.utils import ms2vec, vec2ms
    
    nist_smiles = np.array(json.load(open('DeepEI/data/all_smiles.json')))
    nist_masses = np.load('DeepEI/data/molwt.npy')
    nist_spec = load_npz('DeepEI/data/peakvec.npz').todense()
    
    data = msp.read('E:/data/GCMS DB_AllPublic-KovatsRI-VS2.msp')
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
        try:
            std_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        except:
            std_smi = ''
        specr = spec[i] # true spectrum
        mass = molwt[i] # true mol weight
        try:
            writeSDF(smi, 'Temp/mol.sdf')
        except:
            continue
        cwd = 'E:\\project\\deep-molecular-massspec'
        cmd = 'python make_spectra_prediction.py --input_file=E:/project/DeepEI/Temp/mol.sdf --output_file=E:/project/DeepEI/Temp/mol_anno.sdf --weights_dir=retrain/models/output'
        subprocess.call(cmd, cwd=cwd) # predict spectrum with neims
        try:
            pred_speci = parser_NEIMS('Temp/mol_anno.sdf')
            pred_vec = ms2vec(pred_speci['mz'], pred_speci['intensity']) # spectrum to vector
            os.unlink('Temp/mol_anno.sdf')
        except:
            continue
        
        candidate = np.where(np.abs(nist_masses - mass) < 5)[0] # candidate of nist
        cand_smi = nist_smiles[candidate]
        cand_spec = nist_spec[candidate,:]
        rep_ind = np.where(cand_smi == std_smi)[0] # if the compound in nist, remove it.
        candidate = np.delete(candidate, rep_ind)
        
        # predict spectrum
        mz, intensity = pred_speci['mz'], pred_speci['intensity']  
        intensity /= max(intensity)
        pred_vec = ms2vec(mz, intensity)
        
        # nist spectrum
        nist_vec = np.squeeze(cand_spec[rep_ind[0],:].tolist())
        mz_1, intensity_1 = vec2ms(nist_vec)
        
        # massbank spectrum
        mz_2, intensity_2 = vec2ms(specr)
        
        # compare pred and nist
        plt.vlines(mz, np.zeros(len(mz)), intensity, color='red')
        plt.vlines(mz_1, np.zeros(len(mz_1)), -intensity_1, color='blue')
        weitht_dot_product(pred_vec, nist_vec)
        
        # compare pred and massbank
        plt.vlines(mz, np.zeros(len(mz)), intensity, color='red')
        plt.vlines(mz_2, np.zeros(len(mz_2)), -intensity_2, color='blue')   
        weitht_dot_product(pred_vec, specr)
        
        true_score = weitht_dot_product(specr, pred_vec) # score of true compound
        cand_score = get_score(specr, nist_spec[candidate,:], m='wdp') # score of nist candidates
        rank = len(np.where(cand_score > true_score)[0]) + 1 # rank
        pred_spec.append(pred_vec)
        os.unlink('Temp/mol.sdf')
        
        output.loc[len(output)] = [smi, mass, true_score, rank]
        output.to_csv('rank_neims_massbank.csv')
    pred_spec = np.array(pred_spec)
    np.save('Data/neims_spec_massbank.npy', pred_spec)
    
