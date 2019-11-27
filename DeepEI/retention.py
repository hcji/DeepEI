# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 07:47:48 2019

@author: hcji
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D, MaxPooling1D, concatenate
from tensorflow.keras import metrics, optimizers
from keras.callbacks.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
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
    
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('Model/RI/' + save_name + '_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')
    
    model = Model(layer_in, outputs = layer_output) 
    opt = optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss='mse', metrics=[metrics.mae])
    history = model.fit(X_train, Y_train, epochs=20, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], batch_size=1024, validation_split=0.11)
    '''
    # plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.ylabel('values')
    plt.xlabel('epoch')
    plt.legend(['loss', 'mae', 'val_loss', 'val_mae'], loc='upper left')
    # plt.show()
    plt.savefig("Result/retention_" + save_name + '_loss.png')
    '''
    # predict
    model = load_model('Model/RI/' + save_name + '_model.h5')
    Y_predict = model.predict(X_test)
    r2 = round(r2_score(Y_predict, Y_test), 4)
    mae = round(mean_absolute_error(Y_predict, Y_test), 4)
    '''
    plt.cla()
    plt.plot(Y_test, Y_predict, '.', color = 'blue')
    plt.plot([0,4500], [0,4500], color ='red')
    plt.ylabel('Predicted RI')
    plt.xlabel('Experimental RI')        
    plt.text(0, 4000, 'R2='+str(r2), fontsize=12)
    plt.text(0, 3600, 'MAE='+str(mae), fontsize=12)
    # plt.show()
    plt.savefig("Result/retention_" + save_name + '_r2.png')
    plt.close('all')
    '''
    return {'r2': r2, 'mae': mae}


def build_RI_model_CNN(smiles, RI, method, save_name):
    words = get_dict(smiles, save_path='Model/RI/' + save_name + '_dict.json')
    keep = np.where(~ np.isnan(RI))[0]
    X = []
    Y = []
    for i, smi in enumerate(smiles):
        if i not in keep:
            continue
        xi = one_hot_coding(smi, words, max_len=600)
        if xi is not None:
            X.append(xi.todense())
            Y.append(RI[i])
    X = np.array(X)
    Y = np.array(Y)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
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
        
    for i in range(1):
        layer_dense = Dense(32, activation="relu", kernel_initializer='normal')(layer_dense)
    layer_output = Dense(1, activation="linear", name="output")(layer_dense)
    
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('Model/RI/' + save_name + '_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')
    
    model = Model(layer_in, outputs = layer_output) 
    opt = optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss='mse', metrics=[metrics.mae])
    history = model.fit(X_train, Y_train, epochs=20, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.11)
    '''
    # plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.ylabel('values')
    plt.xlabel('epoch')
    plt.legend(['loss', 'mae', 'val_loss', 'val_mae'], loc='upper left')
    # plt.show()
    plt.savefig("Result/retention_" + save_name + '_loss.png')
    '''
    # predict
    model = load_model('Model/RI/' + save_name + '_model.h5')
    Y_predict = model.predict(X_test)
    r2 = round(r2_score(Y_predict, Y_test), 4)
    mae = round(mean_absolute_error(Y_predict, Y_test), 4)
    '''
    plt.cla()
    plt.plot(Y_test, Y_predict, '.', color = 'blue')
    plt.plot([0,4500], [0,4500], color ='red')
    plt.ylabel('Predicted RI')
    plt.xlabel('Experimental RI')        
    plt.text(0, 4000, 'R2='+str(r2), fontsize=12)
    plt.text(0, 3600, 'MAE='+str(mae), fontsize=12)
    # plt.show()
    plt.savefig("Result/retention_" + save_name + '_r2.png')
    plt.close('all')
    '''
    return {'r2': r2, 'mae': mae}        

            
if __name__ == '__main__':
    
    from scipy.sparse import load_npz
    
    smiles = json.load(open('Data/All_smiles.json'))
    with open('Data/split.json', 'r') as js:
        keep = np.array(json.load(js)['keep'])


    smiles = np.array(json.load(open('Data/All_smiles.json')))[keep]
    rindex = np.load('Data/RI_data.npy')[keep,:]

    morgan = load_npz('Data/Morgan_fp.npz')
    morgan = morgan.todense()[keep,:]
    cdkdes = np.load('Data/CDK_des.npy')[keep,:]

    # remove descriptors includes nan
    keep1 = []
    for i in range(cdkdes.shape[1]):
        v = list(cdkdes[:,i])
        if np.isnan(np.min(v)):
            continue
        else:
            keep1.append(i)
    cdkdes = cdkdes[:, keep1]


    # check the number of data
    n_SimiStdNP = len(np.where(~ np.isnan(rindex[:,0]))[0])
    n_StdNP = len(np.where(~ np.isnan(rindex[:,1]))[0])
    n_StdPolar = len(np.where(~ np.isnan(rindex[:,2]))[0])

    m1 = build_RI_model_descriptor(morgan, cdkdes, rindex[:,0], 'morgan', 'SimiStdNP_DNN_morgan')  
    m2 = build_RI_model_descriptor(morgan, cdkdes, rindex[:,2], 'morgan', 'StdPolar_DNN_morgan')
    m3 = build_RI_model_descriptor(morgan, cdkdes, rindex[:,0], 'descriptor', 'SimiStdNP_DNN_descriptor')
    m4 = build_RI_model_descriptor(morgan, cdkdes, rindex[:,2], 'descriptor', 'StdPolar_DNN_descriptor')
    m5 = build_RI_model_descriptor(morgan, cdkdes, rindex[:,0], 'all', 'SimiStdNP_DNN_all')
    m6 = build_RI_model_descriptor(morgan, cdkdes, rindex[:,2], 'all', 'StdPolar_DNN_all')
    m7 = build_RI_model_CNN(smiles, rindex[:,0], 'single_channel', 'SimiStdNP_CNN_single')
    m8 = build_RI_model_CNN(smiles, rindex[:,2], 'single_channel', 'StdPolar_CNN_single')
    m9 = build_RI_model_CNN(smiles, rindex[:,0], 'multi_channel', 'SimiStdNP_CNN_multi')
    m10 = build_RI_model_CNN(smiles, rindex[:,2], 'multi_channel', 'StdPolar_CNN_multi')
    
    m11 = build_RI_model_descriptor(morgan, cdkdes, rindex[:,1], 'morgan', 'StdNP_DNN_morgan')
    m12 = build_RI_model_descriptor(morgan, cdkdes, rindex[:,1], 'descriptor', 'StdNP_DNN_descriptor')
    m13 = build_RI_model_descriptor(morgan, cdkdes, rindex[:,1], 'all', 'StdNP_DNN_descriptor')
    m14 = build_RI_model_CNN(smiles, rindex[:,1], 'single_channel', 'StdNP_CNN_single')
    m15 = build_RI_model_CNN(smiles, rindex[:,1], 'multi_channel', 'StdNP_CNN_multi')

    output = pd.DataFrame(columns= ['model', 'column', 'mae', 'r2'])
    output.loc[0] = ['DNN_morgan', 'SimiStdNP', m1['mae'], m1['r2']]
    output.loc[1] = ['DNN_morgan', 'StdPolar', m2['mae'], m2['r2']]
    output.loc[2] = ['DNN_descriptor', 'SimiStdNP', m3['mae'], m3['r2']]
    output.loc[3] = ['DNN_descriptor', 'StdPolar', m4['mae'], m4['r2']]
    output.loc[4] = ['DNN_all', 'SimiStdNP', m5['mae'], m5['r2']]
    output.loc[5] = ['DNN_all', 'StdPolar', m6['mae'], m6['r2']]
    output.loc[6] = ['CNN_single', 'SimiStdNP', m7['mae'], m7['r2']]
    output.loc[7] = ['CNN_single', 'StdPolar', m8['mae'], m8['r2']]
    output.loc[8] = ['CNN_multi', 'SimiStdNP', m9['mae'], m9['r2']]
    output.loc[9] = ['CNN_multi', 'StdPolar', m10['mae'], m10['r2']]
    output.loc[10] = ['DNN_morgan', 'StdNP', m11['mae'], m11['r2']]
    output.loc[11] = ['DNN_descriptor', 'StdNP', m12['mae'], m12['r2']]
    output.loc[12] = ['DNN_all', 'StdNP', m13['mae'], m13['r2']]
    output.loc[13] = ['CNN_single', 'StdNP', m14['mae'], m14['r2']]
    output.loc[14] = ['CNN_multi', 'StdNP', m15['mae'], m15['r2']]
    output.to_csv('Result/retention.csv')            