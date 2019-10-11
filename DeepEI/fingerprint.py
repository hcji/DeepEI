# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 07:47:48 2019

@author: hcji
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten, Conv1D, MaxPooling1D, concatenate
from keras import metrics, optimizers
from keras.callbacks.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from sklearn.cross_decomposition import PLSRegression

def build_FP_model_DNN(spec, cdk_fp, i):
    X = spec
    Y = cdk_fp[:,i].flatten()
    # check bias
    frac = np.sum(Y) / max(Y.shape)
    if (frac < 0.1) or (frac > 0.9):
        return None
    
    # label to class
    class_Y = np.vstack((Y, (1-Y))).transpose()
    
    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, class_Y, test_size=0.1)
    
    # build model
    layer_in = Input(shape=(X.shape[1],), name="raw_ms")
    layer_dense = layer_in
    n_nodes = X.shape[1]
    for j in range(3):
        layer_dense = Dense(int(n_nodes), activation="relu")(layer_dense)
        n_nodes = n_nodes * 0.5
    layer_output = Dense(2, activation="softmax", name="output")(layer_dense)
    opt = optimizers.Adam(lr=0.001)
    model = Model(layer_in, layer_output)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # call back
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('Model/Fingerprint/' + str(i) + '.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')
    
    # fit model
    model.fit(X_train, Y_train, epochs=10, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.1)

    # test model
    model = load_model('Model/Fingerprint/' + str(i) + '.h5')
    Y_pred = model.predict(X_test)
    Y_pred = np.round(Y_pred[:,0])
    Y_test = np.round(Y_test[:,0])
    f1 = f1_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    accuracy = accuracy_score(Y_test, Y_pred)
    
    # save model
    model_json = model.to_json()
    with open('Model/Fingerprint/model.json', "w") as js:  
        js.write(model_json)
    model.save_weights('Model/Fingerprint/' + str(i) + '.h5')
    
    return {'frac': frac, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'model': model}


def build_FP_models(spec, cdk_fp, method='DNN', check=True):
    if check: # check if models existed, in case of involuntary interrupt.
        existed = os.listdir('Model/Fingerprint')
        existed = [s.split('.')[0] for s in existed if 'h5' in s]
        try:
            output = pd.read_csv('fingerprint_model.csv', index_col = 0)
        except:
            output = pd.DataFrame(columns=['ind', 'frac', 'accuracy', 'precision', 'recall', 'f1'])
    else:
        output = pd.DataFrame(columns=['ind', 'frac', 'accuracy', 'precision', 'recall', 'f1'])
    functions = {'DNN': build_FP_model_DNN, 'PLSDA': build_FP_model_PLSDA}
    
    for i in tqdm(range(cdk_fp.shape[1])):
        if check:
            if str(i) in existed:
                continue
        res = functions[method](spec, cdk_fp, i)
        if res is None:
            continue
        else:
            output.loc[len(output)] = [i, res['frac'], res['accuracy'], res['precision'], res['recall'], res['f1']]
        del(res)
        output.to_csv('fingerprint_model.csv')
    return output


def build_FP_model_PLSDA(spec, cdk_fp, i, ncomps=range(2,10)):
    X = spec
    Y = cdk_fp[:,i].flatten()
    # check bias
    frac = np.sum(Y) / max(Y.shape)
    if (frac < 0.1) or (frac > 0.9):
        return None
    
    # label to class
    class_Y = np.vstack((Y, (1-Y))).transpose()
    
    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, class_Y, test_size=0.1)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.11)
        
    # train and select best model
    model = None
    best_acc = -1
    Y_valid = np.round(Y_valid[:,0])
    for n in ncomps:
        plsda = PLSRegression(n_components=n).fit(X=X_train,Y=Y_train)
        Y_valhat = plsda.predict(X_valid)
        Y_valhat = np.array([int(pos > neg) for (pos, neg) in Y_valhat])
        Y_val_acc = accuracy_score(Y_valid, Y_valhat)
        if Y_val_acc > best_acc:
            best_acc = Y_val_acc
            model = plsda
    
    # test model
    Y_pred = model.predict(X_test)
    Y_pred = np.array([int(pos > neg) for (pos, neg) in Y_pred])
    Y_test = np.round(Y_test[:,0])
    f1 = f1_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    accuracy = accuracy_score(Y_test, Y_pred)    
    return {'frac': frac, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'model': model}


if __name__ == '__main__':
    
    import json
    from scipy.sparse import load_npz
    
    smiles = json.load(open('Data/All_smiles.json'))
    with open('Data/split.json', 'r') as js:
        keep = np.array(json.load(js)['keep'])
    spec = load_npz('Data/Peak_data.npz')
    cdk_fp = load_npz('Data/CDK_fp.npz')
    
    spec = spec.todense()[keep,:]
    cdk_fp = cdk_fp.todense()[keep,:]
    output = build_FP_models(spec, cdk_fp, method='DNN', check=True)
    output.to_csv('fingerprint_model.csv')