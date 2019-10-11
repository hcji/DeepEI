# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:30:43 2019

@author: hcji
"""

if __name__ == '__main__':
    
    import json
    import numpy as np

    smiles = json.load(open('Data/All_smiles.json'))
    RI = np.load('Data/RI_data.npy')
    withRI = np.where(~np.isnan(RI[:,0]))[0]

    isolate = list(np.random.choice(withRI, 1000))
    keep = [i for i in range(len(smiles)) if i not in isolate]
    isolate = [int(i) for i in isolate]
    keep = [int(i) for i in keep]
    split = {'isolate': isolate, 'keep': keep}
    with open('Data/split.json', 'w') as js:
        json.dump(split, js)