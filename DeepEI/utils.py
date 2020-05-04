# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:43:18 2019

@author: hcji
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from io import StringIO
from tqdm import tqdm
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as numpy2ri
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from sklearn.metrics import jaccard_similarity_score
# jaccard_score funcion return different result. not know why.
# scikit-learn 0.21.2
jaccard_score = jaccard_similarity_score
numpy2ri.activate()
robjects.r('''source('DeepEI/rcdk.R')''')
get_fingerprint = robjects.globalenv['get_fingerprint']
get_descriptors = robjects.globalenv['get_descriptors']
# from pycdk.pycdk import MolFromSmiles, getFingerprint

def ms2vec(peakindex, peakintensity, maxmz=2000):
    output = np.zeros(maxmz)
    for i, j in enumerate(peakindex):
        if round(j) >= maxmz:
            continue
        output[int(round(j))] = float(peakintensity[i])
    output = output / (max(output) + 10 ** -6)
    return output

def vec2ms(vec):
    mz = np.where(vec > 0)[0]
    intensity = vec[mz]
    return mz, intensity

def fp2vec(fp, nbit=6931):
    output = np.zeros(nbit)
    for i in fp:
        output[int(i)] = 1
    return output

def get_cdk_fingerprints(smi):
    types=['standard', 'pubchem', 'kr', 'maccs', 'estate', 'circular']
    fps = []
    for tp in types:
        fps += list(get_fingerprint(smi, tp))
    return fps

def get_cdk_descriptors(smi):
    dsp = list(get_descriptors(smi))
    return dsp

def dot_product(a, b):
    a = np.squeeze(np.asarray(a))
    b = np.squeeze(np.asarray(b))
    return np.dot(a,b)/ np.sqrt((np.dot(a,a)* np.dot(b,b)))

def weitht_dot_product(a, b):
    a = np.squeeze(np.asarray(a))
    b = np.squeeze(np.asarray(b))
    w = np.arange(len(a)) + 1
    wa = np.sqrt(a) * w
    wb = np.sqrt(b) * w
    return np.dot(wa,wb) / np.sqrt((np.dot(wa,wa)* np.dot(wb,wb)))

def get_score(x, X, m='wdp'):
    if m == 'dp':
        s = [dot_product(x, X[i,:]) for i in range(X.shape[0])]
    else:
        s = [weitht_dot_product(x, X[i,:]) for i in range(X.shape[0])]
    return s

def get_ri_score(ri, ris):
    score = []
    if np.isnan(ri):
        return np.zeros(len(ris))
    else:
        dif = np.abs((ris - ri) / 1000)
        score = np.exp(-dif)[:,0]
        return np.array(score)

def get_fp_score(fp, all_fps):
    scores = np.zeros(all_fps.shape[0])
    for i in range(all_fps.shape[0]):
        fpi = all_fps[i,:]
        fpi = fpi.transpose()
        scores[i] = jaccard_score(fp, fpi)
        # scores[i] = jaccard_score(fp, fpi, sample_weight = weights)
    return scores

def writeSDF(smiles, file):
    f = open(file, 'w')
    for smi in tqdm(smiles):
        m = Chem.MolFromSmiles(smi)
        try:
            CalcExactMolWt(m)
        except:
            continue
        sio = StringIO()
        w = Chem.SDWriter(sio)
        w.write(m)
        w=None
        string = sio.getvalue()
        f.write(string)
        
        
def parser_NEIMS(sdf):
    with open(sdf) as t:
        cont = t.readlines()
    all_spectra = []

    start = 0
    spec = pd.DataFrame(columns=['mz', 'intensity'])
    for l in tqdm(cont):
        l = l.replace('\n', '')
        if '$$$$' in l:
            start = 0
            all_spectra.append(spec)
            spec = pd.DataFrame(columns=['mz', 'intensity'])
        if start == 1:
            if l != '':
                l = l.split(' ')
                spec.loc[len(spec)] = np.array(l).astype(int)
        if 'SPECTRUM' in l:
            start = 1
    return all_spectra