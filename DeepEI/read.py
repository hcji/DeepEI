# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 08:24:41 2019

@author: hcji
"""

import sqlite3
import json
import itertools
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from pycdk.pycdk import MolFromSmiles, parser_formula, MolToFormula, getMolecularDescriptor
from DeepEI.utils import ms2vec, fp2vec, get_cdk_fingerprints

spec_path ='NIST2017/NIST_Spec.db'
mol_path ='NIST2017/NIST_Mol.db'

spec_db = sqlite3.connect(spec_path)
spec_cur = spec_db.cursor()
mol_db = sqlite3.connect(mol_path)
mol_cur = mol_db.cursor()

all_mol = mol_cur.execute("select * from catalog")
all_mol = mol_cur.fetchall()

def read_mol(i):
    name = all_mol[i][0]
    smiles = all_mol[i][1]
    spec = spec_cur.execute("select * from catalog where name='%s'" % name)
    spec = spec_cur.fetchall()
    retention = spec[0][1]
    peakindex = json.loads(spec[0][2])
    peakintensity = json.loads(spec[0][3])
    RI = {}
    RI['SemiStdNP'] = np.nan
    RI['StdNP'] = np.nan
    RI['StdPolar'] = np.nan
    if retention != '':
        retention = retention.split(' ')
        for r in retention:
            if 'SemiStdNP' in r:
                RI['SemiStdNP'] = float(r.split('=')[1].split('/')[0])
            if 'StdNP' in r:
                RI['StdNP'] = float(r.split('=')[1].split('/')[0])
            if 'StdPolar' in r:
                RI['StdPolar'] = float(r.split('=')[1].split('/')[0])
    output = {'name': name, 'smiles': smiles, 'RI': RI, 'peakindex': peakindex, 'peakintensity': peakintensity}
    return output


def collect():
    all_smiles = []
    Peak_data = []
    RI_data = []
    Morgan_fp = []
    CDK_fp = []
    CDK_des = []
    Derdive = []
    # for i in tqdm(range(20)):
    for i in tqdm(range(len(all_mol))):
        try:
            m = read_mol(i)
        except:
            continue
        if  'TMS derivative' in m['name']:
            derive = 1
        else:
            derive = 0
        try:
            mol = Chem.MolFromSmiles(m['smiles'])
            smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
            # check element
            elements = parser_formula(MolToFormula(MolFromSmiles(smiles)))
            for e in elements:
                if e not in ['C', 'H', 'O', 'N', 'S', 'P', 'Si', 'F', 'Cl', 'Br', 'I']:
                    print ('contain uncommon element')
                    continue
            morgan_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096))
            cdk_fp = get_cdk_fingerprints(smiles)
            cdk_fp = fp2vec(cdk_fp)
            cdk_des = getMolecularDescriptor(MolFromSmiles(smiles)).values()
            cdk_des  = list(itertools.chain(*cdk_des))
            ri = list(m['RI'].values())
            peak_vec = ms2vec(m['peakindex'], m['peakintensity'])
        except:
            continue
        
        all_smiles.append(smiles)
        Peak_data.append(peak_vec)
        RI_data.append(ri)
        Morgan_fp.append(morgan_fp)
        CDK_fp.append(cdk_fp)
        CDK_des.append(cdk_des)
        Derdive.append(derive)
        
    # save
    np.save('Data/Peak_data.npy', np.array(Peak_data))
    np.save('Data/RI_data.npy', np.array(RI_data))
    np.save('Data/Morgan_fp.npy', np.array(Morgan_fp))
    np.save('Data/CDK_fp.npy', np.array(CDK_fp))
    np.save('Data/CDK_des.npy', np.array(CDK_des))
    np.save('Data/Derdive.npy', np.array(Derdive))
    with open('Data/All_smiles.json', 'w') as t:
        json.dump(all_smiles, t)
  