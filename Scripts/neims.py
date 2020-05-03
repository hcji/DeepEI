# -*- coding: utf-8 -*-
"""
Created on Sat May  2 08:56:20 2020

@author: hcji
"""

import json
import subprocess
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm
from DeepEI.utils import writeSDF
from DeepEI.utils import ms2vec, parser_NEIMS, get_cdk_fingerprints
from libmetgem import msp
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

with open('DeepEI/data/split.json', 'r') as js:
    split = json.load(js)
keep = np.array(split['keep'])
isolate = np.array(split['isolate'])

# NEIMS spectra of NIST
nist_smiles = np.array(json.load(open('DeepEI/data/all_smiles.json')))

writeSDF(nist_smiles, 'Temp/mol.sdf')
cwd = 'E:\\project\\deep-molecular-massspec'
cmd = 'python make_spectra_prediction.py --input_file=E:/project/DeepEI/Temp/mol.sdf --output_file=E:/project/DeepEI/Temp/mol_anno.sdf --weights_dir=model/massspec_weights'
subprocess.call(cmd, cwd=cwd)
spectra = parser_NEIMS('Temp/mol_anno.sdf')

spec_vecs = []
for spec in tqdm(spectra):
    spec_vecs.append(ms2vec(spec['mz'], spec['intensity']))
spec_vecs = np.array(spec_vecs)
spec_vecs1 = csr_matrix(spec_vecs)
save_npz('DeepEI/data/neims_spec_nist.npz', spec_vecs1)


# NEIMS spectra of MassBank
exist_smiles = nist_smiles[keep]
data = msp.read('E:/data/GCMS DB_AllPublic-KovatsRI-VS2.msp')
msbk_smiles = []
msbk_spec = []
msbk_masses = []
for i, (param, ms) in enumerate(tqdm(data)):
    smi = param['smiles']
    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except:
        smi = smi
    try:
        mass = CalcExactMolWt(Chem.MolFromSmiles(smi))
    except:
        continue
    msbk_masses.append(mass)
    msbk_smiles.append(smi)
    msbk_spec.append(ms2vec(ms[:,0], ms[:,1]))

pred_smiles = []
for smi in msbk_smiles:
    if smi in exist_smiles:
        continue
    else:
        pred_smiles.append(smi)
writeSDF(pred_smiles, 'Temp/mol.sdf')
cwd = 'E:\\project\\deep-molecular-massspec'
cmd = 'python make_spectra_prediction.py --input_file=E:/project/DeepEI/Temp/mol.sdf --output_file=E:/project/DeepEI/Temp/mol_anno.sdf --weights_dir=model/massspec_weights'
subprocess.call(cmd, cwd=cwd)
spectra = parser_NEIMS('Temp/mol_anno.sdf')

mols = Chem.SDMolSupplier('Temp/mol_anno.sdf')
pred_smiles = []
pred_masses = []
for m in mols:
    pred_smiles.append(Chem.MolToSmiles(m))
    pred_masses.append(CalcExactMolWt(m))

spec_vecs = []
for spec in tqdm(spectra):
    spec_vecs.append(ms2vec(spec['mz'], spec['intensity']))
spec_vecs = np.array(spec_vecs)
spec_vecs1 = csr_matrix(spec_vecs)

msbk_spec = np.array(msbk_spec)
msbk_spec = csr_matrix(msbk_spec)

save_npz('DeepEI/data/neims_spec_msbk.npz', spec_vecs1)
save_npz('DeepEI/data/msbk_spec.npz', msbk_spec)
with open('DeepEI/data/msbk_smiles.json', 'w') as t:
    json.dump(msbk_smiles, t)
with open('DeepEI/data/neims_msbk_smiles.json', 'w') as t:
    json.dump(pred_smiles, t) 
np.save('DeepEI/data/msbk_masses.npy', msbk_masses)
np.save('DeepEI/data/neims_msbk_masses.npy', pred_masses)

neims_msbk_cdkfps = []
for smi in tqdm(pred_smiles):
    try:
        fp = get_cdk_fingerprints(smi)
    except:
        fp = np.zeros(8034)
    neims_msbk_cdkfps.append(fp)
neims_msbk_cdkfps = np.array(neims_msbk_cdkfps)
neims_msbk_cdkfps = csr_matrix(np.array(neims_msbk_cdkfps))
save_npz('DeepEI/data/neims_msbk_cdkfps', neims_msbk_cdkfps)