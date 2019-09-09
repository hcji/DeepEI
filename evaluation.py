# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:53:52 2019

@author: hcji
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.spatial.distance import jaccard
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from DeepEI.searchsim import get_simcomps
from DeepEI.predict import predict_fingerprint, predict_RI, predict_MS
from DeepEI.utils import get_cdk_fingerprints


with open('Data/split.json', 'r') as js:
    j = json.load(js)
    keep = np.array(j['keep'])
    test = np.array(j['isolate'])
    
smiles = np.array(json.load(open('Data/All_smiles.json')))[keep]
spec = np.load('Data/Peak_data.npy')[keep,:]

test_smiles = np.array(json.load(open('Data/All_smiles.json')))[test]
test_ri = np.load('Data/RI_data.npy')[test,:]
test_spec = np.load('Data/Peak_data.npy')[test,:]

files = os.listdir('Model/Fingerprint')
rfp = np.array([int(f.split('.')[0]) for f in files if '.h5' in f])
rfp = np.sort(rfp)


def calc_molsim(smi1, smi2):
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
    try:
        output = DataStructs.DiceSimilarity(getfp(smi1), getfp(smi2))
    except:
        output = 0
    return output


def get_spec_score(s, spec):
    score = []
    for ss in spec:
        score.append(pearsonr(ss, s)[0])
    return np.array(score)


def get_ri_score(ri, ris):
    if np.isnan(ri):
        return 0
    dif = np.abs((ris - ri) / 1000)
    score = np.exp(-dif)
    return score
    

def get_fp_score(fp, fps):
    score = []
    for f in fps:
        score.append(1-jaccard(f, fp))
    return np.array(score)


def get_candidates(smi, thres=0.8, db='NIST'):
    if db == 'NIST':
        all_smiles = np.array(list(smiles) + list(test_smiles))
        scores = np.array([calc_molsim(smi, s) for s in all_smiles])
        candidates = all_smiles[scores > thres]
    else:
        candidates = get_simcomps(smi, thres*100)['smiles']
        refine = []
        for s in candidates:
            try:
                rf = Chem.MolToSmiles(Chem.MolFromSmiles(s))
                if '.' not in rf:
                    refine.append(rf)
            except:
                pass
        candidates = refine
    return candidates


ranks = []
sim_scores = []
ref_scores = []
all_pred_cdkfp = predict_fingerprint(test_spec)
for i, smi in enumerate(tqdm(test_smiles)):
    if Chem.MolFromSmiles(smi) is None:
        continue
    s = test_spec[i]
    r = test_ri[i, 0]
    pred_cdkfp = all_pred_cdkfp[i,:]
    compare_scores = get_spec_score(s, spec)
    # in case of ref_smi cannot be parsered or no candidates.
    top5 = np.argsort(-compare_scores)[range(5)]
    for i in range(5):
        idx = top5[i]
        ref_smi = smiles[idx]
        ref_spec_score = compare_scores[idx]
        candidates = get_candidates(ref_smi, thres=ref_spec_score - 0.2, db='NIST')
        if len(candidates) > 0:
            # if there is a candidate, break
            break
    can_morgan = [np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=4096)) for s in candidates]
    can_morgan = np.array(can_morgan)
    can_pri = predict_RI(candidates)
    can_spec = predict_MS(can_morgan)
    can_cdkfp = np.array([np.array(get_cdk_fingerprints(s))[rfp] for s in candidates])
    fp_simi = calc_molsim(smi, ref_smi)
    fp_score = get_fp_score(pred_cdkfp, can_cdkfp)
    ri_score = get_ri_score(r, can_pri)[:,0]
    sp_score = get_spec_score(s, can_spec)
    tot_score = fp_score + ri_score + sp_score
    try:
        wh_true = list(candidates).index(smi)
        rank = len(np.where(tot_score > tot_score[wh_true])[0]) + 1
    except:
        rank = 9999
    ranks.append(rank)
    sim_scores.append(fp_simi)
    ref_scores.append(ref_spec_score)
output = pd.DataFrame({'smiles':test_smiles, 'rank':ranks})
output.to_csv('rank_NIST.csv')