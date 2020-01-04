# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:10:20 2019

@author: hcji
"""

import json
import numpy as np
from rdkit import Chem
from io import StringIO
from DeepEI.utils import vec2ms
        
        
if __name__ == '__main__':
    
    import subprocess
    from tqdm import tqdm
    from scipy.sparse import load_npz
    from rdkit.Chem.inchi import MolToInchiKey
    from rdkit.Chem.Descriptors import rdMolDescriptors
    
    with open('Data/split.json', 'r') as js:
        split = json.load(js)
    keep = np.array(split['keep'])
    isolate = np.array(split['isolate'])
    
    smiles = np.array(json.load(open('Data/All_smiles.json')))[keep]
    masses = np.load('Data/MolWt.npy')[keep]
    spec = load_npz('Data/Peak_data.npz')[keep,:].todense()
    
    with open('Data/neims_train.sdf', 'w') as f:
        pass
    
    # write sdf for neims training
    for i in tqdm(range(len(smiles))):
        smi = smiles[i]
        mass = masses[i]
        mz, intensity = vec2ms(np.squeeze(np.asarray(spec[i])))
        intensity = intensity * 1000
        
        # write mol block
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        sio = StringIO()
        w = Chem.SDWriter(sio)
        w.write(m)
        w=None
        string = sio.getvalue()
        
        # string = string.replace('\n     RDKit          2D\n', 'compound_' + str(i))
        string = string.replace('$$$$\n', '')
        with open('Data/neims_train.sdf', 'a') as f:
            f.write(string)
        
        # write info and peaks
        inchikey = MolToInchiKey(m)
        formula = rdMolDescriptors.CalcMolFormula(m)
        string1 = ''
        string1 += '>  <NAME>\n' + 'compound_' + str(i) + '\n\n'
        string1 += '>  <INCHIKEY>\n' + inchikey + '\n\n'
        string1 += '>  <FORMULA>\n' + formula + '\n\n'
        string1 += '>  <EXACT MASS>\n' + str(mass) + '\n\n'
        string1 += '>  <NUM PEAKS>\n' + str(len(mz)) + '\n\n'
        string1 += '>  <MASS SPECTRAL PEAKS>\n'
        for p in range(len(mz)):
            string1 += str(mz[p]) + ' ' + str(int(intensity[p])) + '\n'
        with open('Data/neims_train.sdf', 'a') as f:
            f.write(string1)
        
        # write end
        with open('Data/neims_train.sdf', 'a') as f:
            f.write('\n$$$$\n')
        
    # call neims for retrain
    cwd = 'E:\\project\\deep-molecular-massspec'
    cmd = 'python make_train_test_split.py --main_sdf_name=E:/project/DeepEI/Data/neims_train.sdf --output_master_dir=retrain/spectra_tf_records'
    subprocess.call(cmd, cwd=cwd)
    
    cmd1 = '''python molecule_estimator.py
            --dataset_config_file=retrain/spectra_tf_records/query_replicates_val_predicted_replicates_val.json
            --train_steps=1000 \\ \
            --model_dir=retrain/models/output --hparams=make_spectra_plots=True
            --alsologtostderr'''
    subprocess.call(cmd1, cwd=cwd)
    