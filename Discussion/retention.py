# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 07:47:48 2019

@author: hcji
"""


import json
import numpy as np
import matplotlib.pyplot as plt
from DeepEI.predict import predict_RI

def probability_distribution(data, bins_interval=50, margin=25):
    bins = range(int(min(data)), int(max(data)) + bins_interval - 1, bins_interval)
    print(len(bins))
    plt.xlim(min(data) - margin, max(data) + margin)
    plt.xlabel('Interval')
    plt.ylabel('Probability')
    prob,left,rectangle = plt.hist(x=data, bins=bins, normed=True, histtype='bar', color=['r'])
    plt.show()


if __name__ == '__main__':
    
    smiles = np.array(json.load(open('Data/All_smiles.json')))
    with open('Data/split.json', 'r') as js:
        split = json.load(js)
    keep = np.array(split['keep'])
    isolate = np.array(split['isolate'])

    true_rindex = np.load('Data/RI_data.npy')[isolate,0]
    pred_rindex = predict_RI(smiles[isolate], mode='SimiStdNP')[:,0]
    diff = np.abs(true_rindex - pred_rindex)
    probability_distribution(diff)
    