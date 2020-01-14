# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:25:33 2019

@author: hcji
"""

import numpy as np 
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics import accuracy_score, jaccard_score
# from sklearn.metrics import accuracy_score, jaccard_similarity_score
# jaccard_score = jaccard_similarity_score
from libmetgem import msp
from DeepEI.utils import ms2vec, vec2ms, get_cdk_fingerprints

def get_fp_score(fp, all_fps):
    scores = np.zeros(all_fps.shape[0])
    for i in range(all_fps.shape[0]):
        fpi = all_fps[i,:]
        fpi = fpi.transpose()
        scores[i] = jaccard_score(fp, fpi)
        # scores[i] = jaccard_score(fp, fpi, sample_weight = weights)
    return scores

if __name__ == '__main__':
    
    import os
    import json
    import pandas as pd
    from tqdm import tqdm
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
    from DeepEI.predict import predict_fingerprint
    
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
    
    # only keep fingerprint with f1 > 0.5
    mlp = pd.read_csv('Fingerprint/results/mlp_result.txt', sep='\t', header=None)
    mlp.columns = ['id', 'accuracy', 'precision', 'recall', 'f1']
    fpkeep = mlp['id'][np.where(mlp['f1'] > 0.5)[0]]
    '''
    files = os.listdir('Fingerprint/mlp_models')
    rfp = np.array([int(f.split('.')[0]) for f in files if '.h5' in f])
    rfp = np.sort(rfp) # the index of the used fingerprint
    '''
    spec = np.array(spec)
    pred_fps = predict_fingerprint(spec, fpkeep) # predict fingerprint of the "unknown"
    
    nist_smiles = np.array(json.load(open('DeepEI/data/all_smiles.json')))
    nist_masses = np.load('DeepEI/data/molwt.npy')
    nist_fps = load_npz('DeepEI/data/fingerprints.npz')
    nist_fps = csr_matrix(nist_fps)[:, fpkeep].todense() # fingerprints of nist compounds
    
    output = pd.DataFrame(columns=['smiles', 'mass', 'fp_score', 'rank'])
    for i in tqdm(range(len(smiles))):
        smi = smiles[i]
        std_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        mass = molwt[i]
        pred_fp = pred_fps[i]
        try:
            true_fp = np.array(get_cdk_fingerprints(std_smi)) # true fingerprint of the "unknown"
        except:
            continue
        true_fp = true_fp[fpkeep]
        true_score = jaccard_score(pred_fp, true_fp)  # score of the true compound

        candidate = np.where(np.abs(nist_masses - mass) < 5)[0] # candidate of nist
        cand_smi = nist_smiles[candidate]
        rep_ind = np.where(cand_smi == std_smi)[0] # if the compound in nist, remove it.
        candidate = np.delete(candidate, rep_ind)

        fp_scores = get_fp_score(pred_fp, nist_fps[candidate, :]) # scores of all candidtates
        rank = len(np.where(fp_scores > true_score)[0]) + 1
        
        output.loc[len(output)] = [smi, mass, true_score, rank]
    output.to_csv('Discussion/MassBank_test/results/DeepEI_massbank.csv')
        