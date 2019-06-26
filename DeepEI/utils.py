# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:43:18 2019

@author: hcji
"""

import numpy as np
from pycdk.pycdk import MolFromSmiles, getFingerprint

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
    '''
    supported:
    'pubchem', 'substructure', 'maccs', 'standard', 'extended', 'graph', 'hybridization', 'estate', 'klekota-roth', 'shortestpath', 'signature', 'circular'
    '''
    mol = MolFromSmiles(smi)
    types=['substructure', 'pubchem', 'klekota-roth', 'maccs']
    fingerprints = [getFingerprint(mol, t) for t in types]
    bits = fingerprints[0]['bits'] + list(fingerprints[0]['nbit'] + np.array(fingerprints[1]['bits']))
    bits = []
    for i, fp in enumerate(fingerprints):
        sumbits = sum([fingerprints[j]['nbit'] for j in range(i)])
        this = [sumbits + j for j in fp['bits']]
        bits += this
    return bits
