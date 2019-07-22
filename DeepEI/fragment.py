# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:41:00 2019

@author: hcji
"""

import json
import numpy as np
import pandas as pd
import keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D, MaxPooling1D, concatenate
from tensorflow.keras import metrics, optimizers
from scipy.stats import pearsonr
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from tqdm import tqdm
from smiles_to_onehot.encoding import get_dict, one_hot_coding

'''
with open('Data/split.json') as js:
    split = json.load(js)

keep = split['keep']
spec = np.load('Data/Peak_data.npy')[keep,:]
smiles = np.array(json.load(open('Data/All_smiles.json')))[keep]
'''

def pearson(y_pred, y_true):
    x = K.flatten(y_true)
    y = K.flatten(y_pred)  
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)


def loss(y_pred, y_true):
    return -K.log(pearson(y_pred, y_true))


def refine_spec(smi, vs):
    mass = round(CalcExactMolWt(Chem.MolFromSmiles(smi)))
    vs /= max(vs)
    pk = np.where(vs > 0)[0]
    pk = pk[pk <= mass]
    vspec = np.repeat(0.0, len(vs))
    vspec[pk] = vs[pk]
    return vspec
    

def build_cnn_model(smiles, spec, method, save_name='cnn_model'):
    words = get_dict(smiles, save_path='Model/' + save_name + '_dict.json')
    '''
    with open('Model/cnn_model_multi_channel_dict.json') as js:
        words = json.load(js)
    '''
    X = []
    Y = []
    for i, smi in enumerate(tqdm(smiles)):
        xi = one_hot_coding(smi, words, max_len=1000)
        if xi is not None:
            X.append(xi.todense())
            y = spec[i]
            Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    
    test = np.random.choice(range(len(X)), int(0.1*len(X)))
    train = np.array([i for i in range(len(X)) if i not in test])
    smi_test = smiles[test]
    X_train = X[train,:]; X_test = X[test,:]
    Y_train = Y[train,:]; Y_test = Y[test,:]
    model = create_model(X_train, Y_train, method)

    Y_pred = model.predict(X_test)
    R2 = []
    for i, smi in enumerate(smi_test):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        y_pred = refine_spec(smi, Y_pred[i])
        r2 = pearsonr(Y_test[i], y_pred)[0]
        R2.append(r2)
        
    model_js = model.to_json()
    with open('Model/Fragment/' + save_name + '.json', "w") as json_file:  
        json_file.write(model_js)
    model.save_weights('Model/Fragment/' + save_name + '_forward.h5')
    R2_mean = str(np.mean(R2)) + '±' + str(np.std(R2))
    return {'R2_mean': R2_mean, 'model': model}


def create_model(X, Y, method):
    layer_in = Input(shape=(X.shape[1:3]), name="smile")
    layer_conv = layer_in
    if method == 'single_channel':
        for i in range(5):
            layer_conv = Conv1D(128, kernel_size=4, activation='relu', kernel_initializer='normal')(layer_conv)
            layer_conv = MaxPooling1D(pool_size=2)(layer_conv)
        layer_dense = Flatten()(layer_conv)
    else:
        layer_conv_k1 = Conv1D(128, kernel_size=3, activation='relu', kernel_initializer='normal')(layer_in)
        layer_conv_k1 = MaxPooling1D(pool_size=2)(layer_conv_k1)
        layer_conv_k2 = Conv1D(128, kernel_size=4, activation='relu', kernel_initializer='normal')(layer_in)
        layer_conv_k2 = MaxPooling1D(pool_size=2)(layer_conv_k2)
        layer_conv_k3 = Conv1D(128, kernel_size=5, activation='relu', kernel_initializer='normal')(layer_in) 
        layer_conv_k3 = MaxPooling1D(pool_size=2)(layer_conv_k3)
        layer_conv_k1 = Conv1D(128, kernel_size=3, activation='relu', kernel_initializer='normal')(layer_conv_k1)
        layer_conv_k1 = MaxPooling1D(pool_size=2)(layer_conv_k1)
        layer_conv_k2 = Conv1D(128, kernel_size=4, activation='relu', kernel_initializer='normal')(layer_conv_k2)
        layer_conv_k2 = MaxPooling1D(pool_size=2)(layer_conv_k2)
        layer_conv_k3 = Conv1D(128, kernel_size=5, activation='relu', kernel_initializer='normal')(layer_conv_k3)
        layer_conv_k3 = MaxPooling1D(pool_size=2)(layer_conv_k3)
        layer_dense_k1 = Flatten()(layer_conv_k1)
        layer_dense_k2 = Flatten()(layer_conv_k2)
        layer_dense_k3 = Flatten()(layer_conv_k3)
        layer_dense = concatenate([layer_dense_k1, layer_dense_k2, layer_dense_k3], axis=-1)
        
    for i in range(3):
        layer_dense = Dense(512, activation="relu", kernel_initializer='normal')(layer_dense)
    layer_output = Dense(Y.shape[1], activation="linear", name="output")(layer_dense)
    model = Model(layer_in, outputs = layer_output) 
    opt = optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss=loss, metrics=[pearson])
    history = model.fit(X, Y, epochs=10, validation_split=0.11)
    return model


def build_dnn_model(spec, smiles, morgan, save_name='dnn_model'):
    X = morgan
    Y = spec
    
    test = np.random.choice(range(len(X)), int(0.1*len(X)))
    train = np.array([i for i in range(len(X)) if i not in test])
    
    smi_test = smiles[test]
    X_train = X[train,:]; X_test = X[test,:]
    Y_train = Y[train,:]; Y_test = Y[test,:]   
    
    layer_in = Input(shape=(X.shape[1], ), name="morgan")
    layer_dense = layer_in
    for i in range(5):
        layer_dense = Dense(512, activation="relu", kernel_initializer='normal')(layer_dense)
    layer_output = Dense(Y.shape[1], activation="linear", name="output")(layer_dense)
    
    model = Model(layer_in, outputs = layer_output) 
    opt = optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss=loss, metrics=[pearson])
    history = model.fit(X_train, Y_train, epochs=10, validation_split=0.11)
    
    Y_pred = model.predict(X_test)
    R2 = []
    for i, smi in enumerate(smi_test):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        y_pred = refine_spec(smi, Y_pred[i])
        r2 = pearsonr(Y_test[i], y_pred)[0]
        R2.append(r2)
        
    model_js = model.to_json()
    with open('Model/Fragment/' + save_name + '.json', "w") as json_file:  
        json_file.write(model_js)
    model.save_weights('Model/Fragment/' + save_name + '_forward.h5')
    R2_mean = str(np.mean(R2)) + '±' + str(np.std(R2))
    return {'R2_mean': R2_mean, 'model': model}


def vec2spec(vspec):
    vspec /= max(vspec)
    mz = np.where(vspec > 0)[0]
    intensity = vspec[mz]
    return pd.DataFrame({'mz': mz, 'intensity': intensity})


def pred_spec(smi, method, model, words):
    if method == 'dnn':
        mol = Chem.MolFromSmiles(smi)
        inp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096)
    else:
        inp = one_hot_coding(smi, words, max_len=1000)
    vspec = model.predict(inp)
    vspec = refine_spec(smi, vspec)
    spec = vec2spec(vspec)
    return spec
    

def plot_ms(spectrum):
    plt.figure(figsize=(6, 4))
    plt.vlines(spectrum['mz'], np.zeros(spectrum.shape[0]), np.array(spectrum['intensity']), 'black') 
    plt.axhline(0, color='black')
    plt.show()


def plot_compare_ms(spectrum1, spectrum2, tol=0.05):
    c_mz = []
    c_int = []
    for i in spectrum1.index:
        diffs = abs(spectrum2['mz'] - spectrum1['mz'][i])
        if min(diffs) < tol:
            c_mz.append(spectrum1['mz'][i])
            c_mz.append(spectrum2['mz'][np.argmin(diffs)])
            c_int.append(spectrum1['intensity'][i])
            c_int.append(-spectrum2['intensity'][np.argmin(diffs)])
    c_spec = pd.DataFrame({'mz':c_mz, 'intensity':c_int}) 
    
    plt.figure(figsize=(6, 6))
    plt.vlines(spectrum1['mz'], np.zeros(spectrum1.shape[0]), np.array(spectrum1['intensity']), 'gray')
    plt.axhline(0, color='black')
    plt.vlines(spectrum2['mz'], np.zeros(spectrum2.shape[0]), -np.array(spectrum2['intensity']), 'gray')
    plt.vlines(c_spec['mz'], np.zeros(c_spec.shape[0]), c_spec['intensity'], 'red')
    plt.xlabel('m/z')
    plt.ylabel('Relative Intensity')
    plt.show()