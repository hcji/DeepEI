# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:30:43 2019

@author: hcji
"""

import json
from rdkit import Chem

all_smiles = json.load(open('DeepEI/data/all_smiles.json'))

# test smiles are from "deep molecular massspec"
with open('DeepEI/data/test_smiles.txt') as ts:    
    test_smiles = ts.readlines()
test_smiles = [s.replace(' ', '') for s in test_smiles]

# standardize test smiles
test_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in test_smiles]

# check test smiles index 
test_index = [i for i in range(len(all_smiles)) if all_smiles[i] in test_smiles]
train_index = [i for i in range(len(all_smiles)) if i not in test_index]

# save split result
split = {'isolate': test_index, 'keep': train_index}
with open('Data/split.json', 'w') as js:
    json.dump(split, js)
    
    