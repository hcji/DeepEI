# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 08:28:32 2019

@author: hcji
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.spatial.distance import jaccard, dice

from DeepEI.searchsim import get_simcomps
from DeepEI.predict import predict_fingerprint, predict_RI, predict_MS
from DeepEI.utils import get_cdk_fingerprints

with open('Data/split.json', 'r') as js:
    j = json.load(js)
    keep = np.array(j['keep'])
    test = np.array(j['isolate']) # never used when training
    
files = os.listdir('Model/Fingerprint')
rfp = np.array([int(f.split('.')[0]) for f in files if '.h5' in f])
rfp = np.sort(rfp) # index of cdk fingerprints with trained moded
    
smiles = np.array(json.load(open('Data/All_smiles.json')))
spec = np.load('Data/Peak_data.npy')
morgan = np.load('Data/Morgan_fp.npy')
ri = np.load('Data/RI_data.npy')
cdk_fp = np.load('Data/CDK_fp.npy')
cdk_fp = cdk_fp[:,rfp] # only keep fingerprints with trained model

all_pred_cdkfp = predict_fingerprint(spec[test])
all_pred_spec = predict_MS(morgan[test])

aug_smiles = list(smiles[keep]) + list(smiles[test]) # fictitious augmented libirary
aug_cdkfp = np.vstack((cdk_fp[keep], cdk_fp[test])) # calculated cdk fingerprints via smiles
aug_spec = np.vstack((spec[keep], all_pred_spec)) # experimental spec plus predicted spec
aug_rt = predict_RI(aug_smiles) # predicted RI of all compounds

# define some functions:
def dot_product(a, b):
    return np.dot(a,b)/ np.sqrt((np.dot(a,a)* np.dot(b,b)))

def get_spec_score(s, spec):
    score = []
    for ss in spec:
        score.append(dot_product(ss, s))
    return np.array(score)

def get_ri_score(ri, ris):
    score = []
    if np.isnan(ri):
        return np.zeros(len(ris))
    else:
        dif = np.abs((ris - ri) / 1000)
        score = np.exp(-dif)[:,0]
        return np.array(score)

def get_fp_score(fp, fps):
    score = []
    for f in fps:
        score.append(1-dice(f, fp))
    return np.array(score)

output = pd.DataFrame(columns=['smiles','ranks','fp_scores','spec_scores','rt_scores','tot_scores','best_scores'])
for a, i in enumerate(tqdm(test)):
    smi = smiles[i] # real smiles, treated as unknwn
    s = spec[i] # experimental spec of the unknown
    r = ri[i, 0] # experimental RI of the unknown
    pred_cdkfp = all_pred_cdkfp[a,:] # predicted FP of the unknown
    
    fp_score = get_fp_score(pred_cdkfp, aug_cdkfp)
    ri_score = get_ri_score(r, aug_rt)
    sp_score = get_spec_score(s, aug_spec)
    
    tot_score = fp_score + sp_score + ri_score
    wh_true = list(aug_smiles).index(smi)
    rank = len(np.where(tot_score > tot_score[wh_true])[0]) + 1
    best = max(tot_score)
    
    output.loc[len(output)] = [smi, rank, fp_score[wh_true], sp_score[wh_true], ri_score[wh_true],
                           tot_score[wh_true], best]
    output.to_csv('NIST_test.csv')