# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:41:41 2019

@author: hcji
"""

import os
import sqlite3
from rdkit import Chem
from tqdm import tqdm
import rpy2.robjects as robjects

rdescriptor_path = '''source('{}')'''.format('rcdk.R')
robjects.r(rdescriptor_path)
get_smi_from_sdf = robjects.globalenv['get_smi_from_sdf']
rewrite_sdf = robjects.globalenv['rewrite_sdf']

mol_db = 'NIST_Mol.db'
spec_db = 'NIST_Spec.db'

mc =  sqlite3.connect(mol_db)
sc = sqlite3.connect(spec_db)
mu = mc.cursor()
su = sc.cursor()

sc.execute('create table catalog (name text primary key, RI text, peakindex text, peakintensity text)')
mc.execute('create table catalog (name text primary key, smiles text, inchi text)')

def smi_from_sdf(sdf):
    rewrite_sdf(sdf)
    mol = Chem.MolFromMolFile('temp.sdf')
    smi = Chem.MolToSmiles(mol)
    return smi

def msp_read(msp_file):
    with open(msp_file, 'r') as msp:
        lines = msp.readlines()
    params = {}
    peakindex = []
    peakintensity = []
    lines = [l.replace('\n', '') for l in lines]
    lines = [l.split(': ') for l in lines]
    for l in lines:
        if len(l) == 2:
            params[l[0]] = l[1]
        else:
            pk = l[0].split('; ')
            pk = [p.split(' ') for p in pk]
            for p in pk:
                if len(p) == 2:
                    peakindex.append(int(p[0]))
                    peakintensity.append(int(p[1]))
    return params, peakindex, peakintensity
    

def insert_mol(sdf_file):
    with open(sdf_file, encoding='UTF-8') as s:
        char = s.readlines()
    name = char[0].replace('\n', '')
    smi = smi_from_sdf(sdf_file)
    inchi = Chem.MolToInchi(Chem.MolFromSmiles(smi))
    try:
        mu.execute("insert into catalog values(?, ?, ?)", (name, smi, inchi))
        mc.commit()
    except:
        pass
    

def insert_spec(msp_file):
    params, peakindex, peakintensity = msp_read(msp_file)
    RI = ''
    name = params['Name']
    if 'Retention_index' in params:
        RI = params['Retention_index']
    try:
        su.execute("insert into catalog values(?, ?, ?, ?)", (name, RI, str(peakindex), str(peakintensity)))
        sc.commit()
    except:
        pass

'''
def insert_RI(msp_file):
    params, peakindex, peakintensity = msp_read(msp_file)
    RI = ''
    name = params['Name']
    if 'Retention_index' in params:
        RI = params['Retention_index']
    try:
        su.execute("update catalog set RI=? where name=?", (RI, name))
        sc.commit()
    except:
        pass

msp_dir = 'E:/NIST_EXPORT_2017/NIST2017/NIST_2017_EXPORT'
msp_dir = [os.path.join(msp_dir, f) for f in os.listdir(msp_dir)]
msp_files = []
for f in msp_dir:
    msp_files += [os.path.join(f, s) for s in os.listdir(f)]
for f in tqdm(msp_files):
    try:
        insert_RI(f)
    except:
        continue        
'''

msp_dir = 'E:/NIST_EXPORT_2017/NIST2017/NIST_2017_EXPORT'
msp_dir = [os.path.join(msp_dir, f) for f in os.listdir(msp_dir)]
msp_files = []
for f in msp_dir:
    msp_files += [os.path.join(f, s) for s in os.listdir(f)]
for f in tqdm(msp_files):
    try:
        insert_spec(f)
    except:
        continue
    
mol_dir = 'E:/NIST_EXPORT_2017/NIST2017/NIST_Mol'
mol_dir = [os.path.join(mol_dir, f) for f in os.listdir(mol_dir)]
mol_files = []
for f in mol_dir:
    mol_files += [os.path.join(f, s) for s in os.listdir(f)]
for f in tqdm(mol_files):
    try:
        insert_mol(f)
    except:
        pass