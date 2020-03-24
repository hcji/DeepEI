# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:48:09 2020

@author: hcji
"""

import numpy as np
import pandas as pd
# from sklearn.metrics import jaccard_score
from sklearn.metrics import jaccard_similarity_score
jaccard_score = jaccard_similarity_score


def get_fp_score(fp, all_fps):
    scores = np.zeros(all_fps.shape[0])
    for i in range(all_fps.shape[0]):
        fpi = all_fps[i,:]
        fpi = fpi.transpose()
        scores[i] = jaccard_score(fp, fpi)
        # scores[i] = jaccard_score(fp, fpi, sample_weight = weights)
    return scores


if __name__ == '__main__':
    
    import json
    from tqdm import tqdm
    from scipy.sparse import load_npz, csr_matrix
    from DeepEI.predict import predict_RI, predict_fingerprint
    
    with open('DeepEI/data/split.json', 'r') as js:
        j = json.load(js)
        keep = np.array(j['keep'])
        test = np.array(j['isolate'])
    all_masses = np.load('DeepEI/data/molwt.npy')
    all_smiles = np.array(json.load(open('DeepEI/data/all_smiles.json')))
    
    test_smiles = all_smiles[test]
    test_masses = all_masses[test]
    test_ri = np.load('DeepEI/data/retention.npy')[test,:]
    test_spec = load_npz('DeepEI/data/peakvec.npz').todense()[test,:]
    
    
    # predict RI
    RIs = predict_RI(all_smiles)[:,0]
    
    i = np.where(~ np.isnan(test_ri[:,0]))[0]
    test_smiles = test_smiles[i]
    test_rindex = test_ri[i,0]
    test_spec = test_spec[i,:]
    test_mass = test_masses[i]
    test = i


    # only keep fingerprint with f1 > 0.5
    mlp = pd.read_csv('Fingerprint/results/mlp_result.txt', sep='\t', header=None)
    mlp.columns = ['id', 'accuracy', 'precision', 'recall', 'f1']
    fpkeep = mlp['id'][np.where(mlp['f1'] > 0.5)[0]]
    
    cdk_fp = load_npz('DeepEI/data/fingerprints.npz')
    cdk_fp = csr_matrix(cdk_fp)[:, fpkeep].todense()
    
    # predict fingerprints via ms
    pred_fp = predict_fingerprint(test_spec, fpkeep)
    
    # rank
    output = pd.DataFrame(columns=['smiles', 'mass', 'true RI', 'predict RI', 'mass filter', 'RI filter', 'mass & RI filter'])
    for i in tqdm(range(len(test))):
        smi = test_smiles[i]
        mass = test_mass[i]
        ri = test_rindex[i]
        pred_fpi = pred_fp[i,:]
        trueindex = np.where(all_smiles == smi)[0][0]
        
        # mass filter
        candidate = np.where(np.abs(all_masses - mass) < 5)[0]
        w_true = np.where(candidate==trueindex)[0]
        if len(w_true)==0:
            rank_mass = 99999
        else:
            fp_scores = get_fp_score(pred_fpi, cdk_fp[candidate, :])
            true_fp_score = fp_scores[w_true[0]]
            rank_mass = len(np.where(fp_scores > true_fp_score)[0]) + 1
        
        # ri filter
        candidate = np.where(np.abs(RIs - ri) < 200)[0]
        w_true = np.where(candidate==trueindex)[0]
        if len(w_true)==0:
            rank_ri = 99999
        else:
            pri = RIs[candidate[w_true]][0]
            fp_scores = get_fp_score(pred_fpi, cdk_fp[candidate, :])
            true_fp_score = fp_scores[w_true[0]]
            rank_ri = len(np.where(fp_scores > true_fp_score)[0]) + 1
        
        # mass & ri filter
        candidate = np.where(np.logical_and(np.abs(RIs - ri) < 200, np.abs(all_masses - mass) < 5))[0]
        w_true = np.where(candidate==trueindex)[0]
        if len(w_true)==0:
            rank_combine = 99999
        else:
            fp_scores = get_fp_score(pred_fpi, cdk_fp[candidate, :])
            true_fp_score = fp_scores[w_true[0]]
            rank_combine = len(np.where(fp_scores > true_fp_score)[0]) + 1        
            
        output.loc[len(output)] = [smi, mass, ri, pri, rank_mass, rank_ri, rank_combine]
        output.to_csv('Discussion/NIST_test/results/DeepEI_nist_RI.csv')
    