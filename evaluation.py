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
from scipy.spatial.distance import jaccard, dice
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
    
smiles = np.array(json.load(open('Data/All_smiles.json')))
spec = np.load('Data/Peak_data.npy')
morgan = np.load('Data/Morgan_fp.npy')
ri = np.load('Data/RI_data.npy')

train_smiles = smiles[keep]
train_spec = spec[keep]
train_morgan = morgan[keep]

files = os.listdir('Model/Fingerprint')
rfp = np.array([int(f.split('.')[0]) for f in files if '.h5' in f])
rfp = np.sort(rfp)

def dot_product(a, b):
    return np.dot(a,b)/ np.sqrt((np.dot(a,a)* np.dot(b,b)))
    

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
        score.append(dot_product(ss, s))
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
        score.append(1-dice(f, fp))
    return np.array(score)


def get_candidates(smi, mfp, thres=0.8, db='NIST'):
    if db == 'NIST':
        scores = get_fp_score(mfp, morgan)
        candidates = smiles[scores>thres]
        morgans = morgan[scores>thres]
    else:
        candidates = get_simcomps(smi, thres*100)['smiles']
        refine = []
        morgans = []
        for s in candidates:
            try:
                rf = Chem.MolToSmiles(Chem.MolFromSmiles(s), kekuleSmiles=True)
                morg = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=4096))
                if '.' in rf:
                    continue
                refine.append(rf)
                morgans.append(morg)
            except:
                continue
        candidates = refine
    return candidates, morgans


out_smiles = []
ranks = []
sim_scores = []
ref_scores = []
all_pred_cdkfp = predict_fingerprint(spec[test])

for a, i in enumerate(tqdm(test)):
    smi = smiles[i]
    s = spec[i]
    r = ri[i, 0]
    pred_cdkfp = all_pred_cdkfp[a,:]
    compare_scores = get_spec_score(s, train_spec)
   
    tops = np.argsort(-compare_scores)[range(5)]
    all_candidates = []
    all_morgans = []
    for idx in tops:
        ref_smi = train_smiles[idx]
        ref_morgan = train_morgan[idx]
        ref_spec_score = compare_scores[idx]
        candidates, morgans = get_candidates(ref_smi, ref_morgan, thres=ref_spec_score - 0.3, db='NIST')
        for aa, cc in enumerate(candidates):
            if cc not in all_candidates:
                all_candidates.append(str(cc))
                all_morgans.append(morgans[aa])
    candidates = np.array(all_candidates)
    can_morgan = np.array(all_morgans)
    if len(candidates) == 0:
        continue
    # can_morgan = [np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=4096)) for s in candidates]
    can_pri = predict_RI(candidates)
    can_spec = predict_MS(can_morgan)
    can_cdkfp = np.array([np.array(get_cdk_fingerprints(s))[rfp] for s in candidates])
    # fp_simi = calc_molsim(smi, ref_smi)
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
    # sim_scores.append(fp_simi)
    ref_scores.append(ref_spec_score)
    out_smiles.append(smi)
    
    output = pd.DataFrame({'smiles':out_smiles, 'rank':ranks, 'ref_scores': ref_scores})
    output.to_csv('rank_NIST.csv')