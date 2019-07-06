# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 07:47:48 2019

@author: hcji
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D, MaxPooling1D, concatenate, Embedding, LSTM
from tensorflow.keras import metrics, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, r2_score
from tqdm import tqdm
from sklearn.cross_decomposition import PLSRegression
from smiles_to_onehot.encoding import get_dict, one_hot_coding


def build_RI_model_descriptor(morgan, cdkdes, RI, descriptor, save_name):
    # remove nan
    keep = np.where(~ np.isnan(RI))[0]
    if descriptor == 'all':
        X = np.hstack((morgan[keep,:], cdkdes[keep,:]))
    elif descriptor == 'morgan':
        X = morgan[keep,:]
    else:
        X = cdkdes[keep,:]
    Y = RI[keep]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    
    # scale
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, 'Model/RI/' + save_name + '_scaler.save') 
    
    # train model
    layer_in = Input(shape=(X.shape[1],), name="morgan_fp")
    layer_dense = layer_in
    n_nodes = X.shape[1]
    for j in range(5):
        layer_dense = Dense(int(n_nodes), activation="relu")(layer_dense)
        n_nodes = n_nodes * 0.5
    layer_output = Dense(1, activation="linear", name="output")(layer_dense)
    model = Model(layer_in, outputs = layer_output) 
    opt = optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss='mse', metrics=[metrics.mae])
    history = model.fit(X_train, Y_train, epochs=20, batch_size=1024, validation_split=0.11)
    
    # plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.ylabel('values')
    plt.xlabel('epoch')
    plt.legend(['loss', 'mae', 'val_loss', 'val_mae'], loc='upper left')
    plt.show()
    
    # predict
    Y_predict = model.predict(X_test)
    r2 = round(r2_score(Y_predict, Y_test), 4)
    mae = round(mean_absolute_error(Y_predict, Y_test), 4)
    plt.cla()
    plt.plot(Y_test, Y_predict, '.', color = 'blue')
    plt.plot([0,4500], [0,4500], color ='red')
    plt.ylabel('Predicted RI')
    plt.xlabel('Experimental RI')        
    plt.text(0, 4000, 'R2='+str(r2), fontsize=15)
    plt.show()
    
    # save model
    model.save('Model/RI/' + save_name + '_model.h5')
    return {'r2': r2, 'mae': mae, 'model': model}


def build_RI_model_CNN(smiles, RI, method, save_name):
    words = get_dict(smiles, save_path='Model/RI/' + save_name + '_dict.json')
    keep = np.where(~ np.isnan(RI))[0]
    X = []
    Y = []
    for i, smi in enumerate(smiles):
        if i not in keep:
            continue
        xi = one_hot_coding(smi, words, max_len=1000)
        if xi is not None:
            X.append(xi.todense())
            Y.append(RI[i])
    X = np.array(X)
    Y = np.array(Y)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    layer_in = Input(shape=(X.shape[1:3]), name="smile")
    layer_conv = layer_in
    if method == 'singe_channel':
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
        
    for i in range(1):
        layer_dense = Dense(32, activation="relu", kernel_initializer='normal')(layer_dense)
    layer_output = Dense(1, activation="linear", name="output")(layer_dense)
    model = Model(layer_in, outputs = layer_output) 
    opt = optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss='mse', metrics=[metrics.mae])
    history = model.fit(X_train, Y_train, epochs=20, validation_split=0.11)

    # plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.ylabel('values')
    plt.xlabel('epoch')
    plt.legend(['loss', 'mae', 'val_loss', 'val_mae'], loc='upper left')
    plt.show()
    
    # predict
    Y_predict = model.predict(X_test)
    r2 = round(r2_score(Y_predict, Y_test), 4)
    mae = round(mean_absolute_error(Y_predict, Y_test), 4)
    plt.cla()
    plt.plot(Y_test, Y_predict, '.', color = 'blue')
    plt.plot([0,4500], [0,4500], color ='red')
    plt.ylabel('Predicted RI')
    plt.xlabel('Experimental RI')        
    plt.text(0, 4000, 'R2='+str(r2), fontsize=15)
    plt.show()
    
    # save model
    model.save('Model/RI/' + save_name + '_model.h5')
    return {'r2': r2, 'mae': mae, 'model': model}        
    

def build_RI_model_RNN(smiles, RI, save_name):
    words = get_dict(smiles, save_path='Model/RI/' + save_name + '_dict.json')
    keep = np.where(~ np.isnan(RI))[0]
    X = []
    Y = []
    for i, smi in enumerate(tqdm(smiles)):
        if i not in keep:
            continue
        xi = one_hot_coding(smi, words, max_len=1000)
        if xi is not None:
            xi = xi.todense()
            xj = []
            for k in xi:
                if np.sum(k) > 0:
                    xj.append(k.argmax())
                else:
                    break
            X.append(np.array(xj))
            Y.append(RI[i])
    # X = np.array(X)
    # Y = np.array(Y)    

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    
    X_train = pad_sequences(X_train, maxlen=1000)
    X_test = pad_sequences(X_test, maxlen=1000)
    
    model = Sequential()
    model.add(Embedding(X_train.shape[0], 300))
    model.add(LSTM(256))
    model.add(Dense(1, activation='linear'))    
    opt = optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss='mse', metrics=[metrics.mae])
    history = model.fit(X_train, Y_train, epochs=20, validation_split=0.11)    

    # plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.ylabel('values')
    plt.xlabel('epoch')
    plt.legend(['loss', 'mae', 'val_loss', 'val_mae'], loc='upper left')
    plt.show()
    
    # predict
    Y_predict = model.predict(X_test)
    r2 = round(r2_score(Y_predict, Y_test), 4)
    mae = round(mean_absolute_error(Y_predict, Y_test), 4)
    plt.cla()
    plt.plot(Y_test, Y_predict, '.', color = 'blue')
    plt.plot([0,4500], [0,4500], color ='red')
    plt.ylabel('Predicted RI')
    plt.xlabel('Experimental RI')        
    plt.text(0, 4000, 'R2='+str(r2), fontsize=15)
    plt.show()
    
    # save model
    model.save('Model/RI/' + save_name + '_model.h5')
    return {'r2': r2, 'mae': mae, 'model': model}      


def build_FP_model_DNN(spec, cdk_fp, i):
    X = spec
    Y = cdk_fp[:,i]
    # check bias
    frac = np.sum(Y) / len(Y)
    if (frac < 0.1) or (frac > 0.9):
        return None
    
    # encoder 
    encoder = LabelEncoder()
    encoder.fit([1, 0])
    encoded_Y = encoder.transform(Y)
    class_Y = keras.utils.to_categorical(encoded_Y, 2)
    
    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, class_Y, test_size=0.1)
    
    # build model
    layer_in = Input(shape=(X.shape[1],), name="raw_ms")
    layer_dense = layer_in
    n_nodes = X.shape[1]
    for j in range(5):
        layer_dense = Dense(int(n_nodes), activation="relu")(layer_dense)
        n_nodes = n_nodes * 0.5
    layer_output = Dense(2, activation="softmax", name="output")(layer_dense)
    opt = optimizers.Adam(lr=0.001)
    model = Model(layer_in, layer_output)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=5, batch_size=1024, validation_split=0.11)
    model.save('Model/Fingerprint/' + str(i) + '.h5')
    '''
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylabel('values')
    plt.xlabel('epoch')
    plt.legend(['acc', 'val_acc'], loc='upper left')
    plt.show()
    '''
    
    # test model
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


def build_FP_models(spec, cdk_fp, method='DNN'):
    existed = os.listdir('Model/Fingerprint')
    existed = [s.split('.')[0] for s in existed if 'h5' in s]
    try:
        output = pd.read_csv('fingerprint_model.csv', index_col = 0)
    except:
        output = pd.DataFrame(columns=['ind', 'frac', 'accuracy', 'precision', 'recall', 'f1'])
    functions = {'DNN': build_FP_model_DNN, 'PLSDA': build_FP_model_PLSDA}
    
    for i in tqdm(range(cdk_fp.shape[1])):
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
    Y = cdk_fp[:,i]
    # check bias
    frac = np.sum(Y) / len(Y)
    if (frac < 0.1) or (frac > 0.9):
        return None
    
    # encoder 
    encoder = LabelEncoder()
    encoder.fit([1, 0])
    encoded_Y = encoder.transform(Y)
    class_Y = keras.utils.to_categorical(encoded_Y, 2)
    
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
    
   