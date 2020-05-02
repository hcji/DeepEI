# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:36:13 2020

@author: hcji
"""

import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from io import StringIO
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz
from DeepEI.utils import ms2vec, writeSDF, parser_NEIMS, get_score

# NEIMS spectra of NIST
nist_smiles = np.array(json.load(open('DeepEI/data/all_smiles.json')))

writeSDF(nist_smiles, 'Temp/mol.sdf')
cwd = 'E:\\project\\deep-molecular-massspec'
cmd = 'python make_spectra_prediction.py --input_file=E:/project/DeepEI/Temp/mol.sdf --output_file=E:/project/DeepEI/Temp/mol_anno.sdf --weights_dir=model/massspec_weights'
subprocess.call(cmd, cwd=cwd)
spectra = parser_NEIMS('Temp/mol_anno.sdf')

spec_vecs = []
for spec in tqdm(spectra):
    spec_vecs.append(ms2vec(spec['mz'], spec['intensity']))
spec_vecs = np.array(spec_vecs)
spec_vecs1 = csr_matrix(spec_vecs)
save_npz('DeepEI/data/neims_spec_nist.npz', spec_vecs1)

# predict spectra of Chebi and HMDB
db_smiles = np.array(json.load(open('DeepEI/data/chebi_smiles.json')) + json.load(open('DeepEI/data/hmdb_smiles.json')))

writeSDF(db_smiles, 'Temp/mol.sdf')
cwd = 'E:\\project\\deep-molecular-massspec'
cmd = 'python make_spectra_prediction.py --input_file=E:/project/DeepEI/Temp/mol.sdf --output_file=E:/project/DeepEI/Temp/mol_anno.sdf --weights_dir=model/massspec_weights'
subprocess.call(cmd, cwd=cwd)
spectra = parser_NEIMS('Temp/mol_anno.sdf')

spec_vecs = []
for spec in tqdm(spectra):
    spec_vecs.append(ms2vec(spec['mz'], spec['intensity']))
spec_vecs = np.array(spec_vecs)
spec_vecs1 = csr_matrix(spec_vecs)
save_npz('DeepEI/data/neims_spec_db.npz', spec_vecs1)

if __name__ == '__main__':
    
    from scipy.sparse import load_npz
    
    with open('DeepEI/data/split.json', 'r') as js:
        j = json.load(js)
        test = np.array(j['isolate'])
    
    exp_spec = load_npz('DeepEI/data/peakvec.npz').todense()
    prd_spec = load_npz('DeepEI/data/neims_spec_nist.npz').todense()

    output = pd.DataFrame(columns=['smiles', 'score', 'rank'])
    for i in tqdm(test):
        smi = nist_smiles[i]
        true_vec = np.squeeze(np.asarray(exp_spec[i,:])) # spectrum from NIST
        pred_vec = np.squeeze(np.asarray(prd_spec[i,:])) # spectrum from NEIMS prediction
        if np.sum(pred_vec) == 0:
            continue
        
        cand_score = get_score(true_vec, prd_spec, m='wdp') # score of nist candidates        
        true_score = cand_score[i]
        rank = len(np.where(cand_score > true_score)[0]) + 1
        output.loc[len(output)] = [smi, true_score, rank]
    output.to_csv('Discussion/NIST_test/results/neims_nist.csv')
