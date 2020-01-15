# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:20:45 2019

@author: hcji
"""

import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score

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
    return np.array(s)

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
    from rdkit import Chem
    from scipy.sparse import load_npz, csr_matrix
    from tqdm import tqdm
    from DeepEI.utils import get_cdk_fingerprints
    from DeepEI.predict import predict_fingerprint
    
    smiles = np.array(json.load(open('DeepEI/data/all_smiles.json')))
    masses = np.load('DeepEI/data/molwt.npy')
    spec = load_npz('DeepEI/data/peakvec.npz').todense()

    with open('DeepEI/data/split.json', 'r') as js:
        j = json.load(js)
        keep = np.array(j['keep'])
        test = np.array(j['isolate'])

    # ms for test
    test_smiles = smiles[test]
    test_mass = masses[test]
    test_spec = spec[test,:]
    
    pred_spec = np.load('Discussion/NIST_test/neims_spec_nist.npy')
    
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
    pred_fps = predict_fingerprint(test_spec, fpkeep) # predict fingerprint of the "unknown"
    cdk_fp = load_npz('DeepEI/data/fingerprints.npz')
    cdk_fp = csr_matrix(cdk_fp)[:, fpkeep].todense()
    
    output = pd.DataFrame(columns=['smiles', 'mass', 'score', 'rank'])
    for i in tqdm(range(len(test))):
        smi = test_smiles[i]
        std_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        mass = test_mass[i]
        speci = test_spec[i]
        pred_fp = pred_fps[i]
        pred_sp = pred_spec[i]
        try:
            true_fp = np.array(get_cdk_fingerprints(smi)) # true fingerprint of the "unknown"
        except:
            continue
        true_fp = true_fp[fpkeep]
        true_score_fp = jaccard_score(pred_fp, true_fp)  # fp score of the true compound
        true_score_sp = weitht_dot_product(speci, pred_sp) # sp score of the true compound
        true_score = 0.5*true_score_fp + 0.5*true_score_sp
        
        candidate = np.where(np.abs(masses - mass) < 5)[0] # candidate of nist
        cand_smi = smiles[candidate]
        rep_ind = np.where(cand_smi == std_smi)[0] # if the compound in nist, remove it.
        candidate = np.delete(candidate, rep_ind)
        
        fp_scores = get_fp_score(pred_fp, cdk_fp[candidate, :]) # scores of all candidtates
        sp_scores = get_score(speci, spec[candidate,:], m='wdp')
        cand_scores = 0.5*fp_scores + 0.5*sp_scores
        
        rank = len(np.where(cand_scores > true_score)[0]) + 1
        
        output.loc[len(output)] = [smi, mass, true_score, rank]
        output.to_csv('Discussion/NIST_test/results/combine_nist.csv')        
