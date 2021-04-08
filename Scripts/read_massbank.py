# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:46:16 2021

@author: hcji
"""


import numpy as np
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from pycdk.pycdk import MolFromSmiles, parser_formula, MolToFormula, getMolecularDescriptor
from DeepEI.utils import ms2vec, fp2vec, get_cdk_fingerprints, get_cdk_descriptors

f = 'd:/MoNA-export-GC-MS_Spectra.sdf'

all_smiles = []
Peak_data = []
RI_data = []
Morgan_fp = []
CDK_fp = []
CDK_des = []
MolWt = []
    
mols = Chem.SDMolSupplier(f)
for m in tqdm(mols):
    if m is None:
        continue
    smi = Chem.MolToSmiles(m)
    morgan_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=4096))
    cdk_fp = get_cdk_fingerprints(smi)
    cdk_des = np.array(get_cdk_descriptors(smi))
    
    props = m.GetPropsAsDict()
    molwt = props['EXACT MASS']
    peaks = props['MASS SPECTRAL PEAKS']
    peaks = peaks.split('\n')
    peakindex = np.array([round(float(p.split(' ')[0])) for p in peaks])
    peakintensity = np.array([float(p.split(' ')[1]) for p in peaks])
    peak_vec = ms2vec(m['peakindex'], m['peakintensity'])

    all_smiles.append(smi)
    Peak_data.append(peak_vec)
    Morgan_fp.append(morgan_fp)
    CDK_fp.append(cdk_fp)
    CDK_des.append(cdk_des)
    MolWt.append(molwt)
