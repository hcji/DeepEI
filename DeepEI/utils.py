# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:43:18 2019

@author: hcji
"""

import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as numpy2ri
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

def fp2vec(fp, nbit=6931):
    output = np.zeros(nbit)
    for i in fp:
        output[int(i)] = 1
    return output

def get_cdk_fingerprints(smi):
    types=['standard', 'pubchem', 'kr', 'maccs']
    fps = []
    for tp in types:
        fps += list(get_fingerprint(smi, tp))
    return fps

def get_cdk_descriptors(smi):
    dsp = list(get_descriptors(smi))
    return dsp