# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:46:45 2019

@author: hcji
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 08:24:41 2019

@author: hcji
"""

import sqlite3
import json
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from pycdk.pycdk import MolFromSmiles, parser_formula, MolToFormula, getMolecularDescriptor
from DeepEI.utils import ms2vec, fp2vec, get_cdk_fingerprints, get_cdk_descriptors

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
            if ('StdNP' in r) and ('Semi' not in r):
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
    MolWt = []
    # for i in tqdm(range(20)):
    for i in tqdm(range(len(all_mol))):
        try:
            m = read_mol(i)
        except:
            continue
        '''
        if  'TMS derivative' in m['name']:
            derive = 1
        else:
            derive = 0
        '''
        try:
            mol = Chem.MolFromSmiles(m['smiles'])
            molwt = CalcExactMolWt(mol)
            if molwt > 2000:
                continue
            smiles = Chem.MolToSmiles(mol)
            # check element
            elements = parser_formula(MolToFormula(MolFromSmiles(smiles)))
            for e in elements:
                if e not in ['C', 'H', 'O', 'N', 'S', 'P', 'Si', 'F', 'Cl', 'Br', 'I']:
                    raise ValueError ('contain uncommon element')
            morgan_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096))
            cdk_fp = get_cdk_fingerprints(smiles)
            # cdk_fp = fp2vec(cdk_fp)
            cdk_des = np.array(get_cdk_descriptors(smiles))
            # cdk_des = getMolecularDescriptor(MolFromSmiles(smiles)).values()
            # cdk_des  = np.array(list(itertools.chain(*cdk_des)))
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
        MolWt.append(molwt)
 
    # save
    np.save('DeepEI/data/retention.npy', np.array(RI_data))
    np.save('DeepEI/data/Data/descriptor.npy', np.array(CDK_des))
    np.save('DeepEI/data/Data/molwt.npy', np.array(MolWt))
    
    Peak_data = csr_matrix(np.array(Peak_data))
    Morgan_fp = csr_matrix(np.array(Morgan_fp))
    CDK_fp = csr_matrix(np.array(CDK_fp))
    save_npz('DeepEI/data/peakvec.npz', Peak_data)
    save_npz('DeepEI/data/morgan.npz', Morgan_fp)
    save_npz('DeepEI/data/fingerprints.npz', CDK_fp)
    
    with open('DeepEI/data/all_smiles.json', 'w') as t:
        json.dump(all_smiles, t)


if __name__ == '__main__':
    collect()