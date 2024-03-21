# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 08:50:45 2024

@author: DELL
"""


import os
import itertools
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import inchi

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from openbabel import pybel
from PyFingerprint.fingerprint import get_fingerprint


# parse and build molecule dict
sdf_path = 'D:/NIST20_Export/sdf_export'
sdf_files = [sdf_path + '/{}'.format(f) for f in os.listdir(sdf_path)]

mol_dict = {}
for f in tqdm(sdf_files):
    for mol in pybel.readfile("sdf", f):
        title = mol.title
        smiles = mol.write("smi").strip().split('\t')[0]
        try:
            inchikey = inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))
        except:
            continue
        mol_dict[title] = {}
        mol_dict[title]['smiles'] = smiles
        mol_dict[title]['inchikey'] = inchikey


# parse and build spectra dict
def parse_msp_file(filename):
    data = {}
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('Name:'):
                current_entry = line.split(': ')[1].strip()
                data[current_entry] = {'mass': [], 'intensities': []}
            elif line[0].isdigit():
                mass_intensities = line.split('; ')
                mass = [int(s.split(' ')[0]) for s in mass_intensities if s != '\n']
                intensity = [int(s.split(' ')[1]) for s in mass_intensities if s != '\n']
                data[current_entry]['mass'] += mass
                data[current_entry]['intensities'] += intensity
    return data


msp_path = 'D:/NIST20_Export/msp_export'
msp_files = [msp_path + '/{}'.format(f) for f in os.listdir(msp_path)]

all_spectra = {}
for f in tqdm(msp_files):
    all_spectra.update(parse_msp_file(f))


# build training data
def ms2vec(peakindex, peakintensity, maxmz=2000):
    output = np.zeros(maxmz)
    for i, j in enumerate(peakindex):
        if round(j) >= maxmz:
            continue
        output[int(round(j))] = float(peakintensity[i])
    output = output / (max(output) + 10 ** -6)
    return output

test_list = open('Reorganize/testset_inchikey.txt', 'r').readlines()
test_list = [s[:14] for s in test_list]

names = list(set(all_spectra.keys()) & set(mol_dict.keys()))
spectra_mat = []
fingerprint_mat = []

types=['standard', 'pubchem', 'klekota-roth', 'maccs', 'estate', 'circular']
for n in tqdm(names):
    s = all_spectra[n]
    smi = mol_dict[n]['smiles']
    key = mol_dict[n]['inchikey'][:14]
    if key in test_list:
        continue
    fp = [get_fingerprint(smi, t).to_numpy() for t in types]
    fp = list(itertools.chain(*fp))
    vec = ms2vec(s['mass'], s['intensities'])
    spectra_mat.append(vec)
    fingerprint_mat.append(fp)
spectra_mat = np.array(spectra_mat)
fingerprint_mat = np.array(fingerprint_mat)

np.save('spectra_mat.npy', spectra_mat)
np.save('fingerprint_mat.npy', fingerprint_mat)

class MLP:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.X_tr, self.X_ts, self.Y_tr, self.Y_ts = train_test_split(X, Y, test_size=0.1)
        
        inp = Input(shape=(X.shape[1],))
        hid = inp
        n = X.shape[1]
        for j in range(3):
            hid = Dense(n, activation="relu")(hid)
            n = int(n * 0.5)
        prd = Dense(2, activation="softmax")(hid)
        opt = optimizers.Adam(lr=0.001)
        model = Model(inp, prd)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
        self.model = model
        
    def train(self, epochs=8):
        self.model.fit(self.X_tr, self.Y_tr, epochs=epochs)
    
    def test(self):
        Y_pred = np.round(self.model.predict(self.X_ts))
        f1 = f1_score(self.Y_ts[:,0], Y_pred[:,0])
        precision = precision_score(self.Y_ts[:,0], Y_pred[:,0])
        recall = recall_score(self.Y_ts[:,0], Y_pred[:,0])
        accuracy = accuracy_score(self.Y_ts[:,0], Y_pred[:,0])
        return accuracy, precision, recall, f1
    
    def save(self, path):
        model_json = self.model.to_json()
        with open('Fingerprint/mlp_models/model.json', "w") as js:  
            js.write(model_json)
        self.model.save_weights(path)
        K.clear_session()


# build model
for i in tqdm(range(fingerprint_mat.shape[1])):
    y = fingerprint_mat[:,i]
    # check: 0.1 < bias < 0.9
    fr = np.sum(y) / len(y)
    if (fr < 0.1) or (fr > 0.9):
        continue
    Y = np.vstack((y, (1-y))).transpose()
    
    mlp_result = open('Fingerprint/results/mlp_result_new.txt', 'a+')
    
    # mlp model
    mlp = MLP(spectra_mat, Y)
    mlp.train()
    mlp_res = mlp.test()
    mlp_result.write("\t".join([str(i)] + [str(j) for j in mlp_res]))
    mlp_result.write("\n")
    mlp.save('Fingerprint/mlp_models/{}.h5'.format(i))






