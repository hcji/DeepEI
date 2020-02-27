# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:22:42 2020

@author: hcji
"""


import numpy as np
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D, MaxPooling1D, concat
from tensorflow.keras import optimizers
from sklearn.metrics import mean_absolute_error, r2_score
from smiles_to_onehot.encoding import get_dict, one_hot_coding

class multi_CNN:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.X_tr, self.X_ts, self.Y_tr, self.Y_ts = train_test_split(X, Y, test_size=0.1)
        
        inp = Input(shape=(X.shape[1:3]))
        n = X.shape[1]

        hidv1 = Conv1D(n, kernel_size=2, activation='relu')(inp)
        hidv1 = MaxPooling1D(pool_size=2)(hidv1)
        hidv1 = Conv1D(n, kernel_size=2, activation='relu')(hidv1)
        hidv1 = MaxPooling1D(pool_size=2)(hidv1)
        hidv1 = Flatten()(hidv1)
        
        hidv2 = Conv1D(n, kernel_size=3, activation='relu')(inp)
        hidv2 = MaxPooling1D(pool_size=3)(hidv2)
        hidv2 = Conv1D(n, kernel_size=3, activation='relu')(hidv2)
        hidv2 = MaxPooling1D(pool_size=3)(hidv2)
        hidv2 = Flatten()(hidv2)
        
        hidv3 = Conv1D(n, kernel_size=4, activation='relu')(inp)
        hidv3 = MaxPooling1D(pool_size=4)(hidv3)
        hidv3 = Conv1D(n, kernel_size=4, activation='relu')(hidv3)
        hidv3 = MaxPooling1D(pool_size=4)(hidv3)
        hidv3 = Flatten()(hidv3)

        hid = concat([hidv1, hidv2, hidv3], axis=-1)
        hid = Dense(32, activation="relu")(hid)
        hid = Dense(32, activation="relu")(hid)
        
        prd = Dense(1, activation="linear")(hid)
        opt = optimizers.Adam(lr=0.001)
        model = Model(inp, prd)
        model.compile(optimizer=opt, loss='mse', metrics=['mae'])
        self.model = model
    
    def train(self, epochs=50):
        history = self.model.fit(self.X_tr, self.Y_tr, epochs=epochs, validation_split = 0.1)
        plt.cla()
        plt.plot(history.history['val_loss'], alpha= 0.8)
        plt.plot(history.history['val_mean_absolute_error'], alpha= 0.8)
        plt.legend(['loss', 'accuracy'], loc="lower left")
        plt.xlabel('epoch')
        return history
    
    def test(self):
        Y_test = self.Y_ts
        Y_pred = np.round(self.model.predict(self.X_ts))
        r2 = round(r2_score(Y_pred, Y_test), 4)
        mae = round(mean_absolute_error(Y_pred, Y_test), 4)

        plt.cla()
        plt.plot(Y_test, Y_pred, '.', color = 'blue')
        plt.plot([0,4500], [0,4500], color ='red')
        plt.ylabel('Predicted RI')
        plt.xlabel('Experimental RI')        
        plt.text(0, 4000, 'R2='+str(r2), fontsize=12)
        plt.text(0, 3600, 'MAE='+str(mae), fontsize=12)
        plt.show()
        return r2, mae
    
    def save(self, path):
        self.model.save(path)
        K.clear_session()
        

if __name__ == '__main__':
    
    import json
    
    with open('DeepEI/data/split.json', 'r') as js:
        keep = np.array(json.load(js)['keep'])
        
    smiles = np.array(json.load(open('DeepEI/data/all_smiles.json')))[keep]
    rindex = np.load('DeepEI/data/retention.npy')[keep,:]
    
    words = get_dict(smiles, save_path='DeepEI/data/words.json')
    smiles = [one_hot_coding(smi, words, max_len=100) for smi in smiles]
    
    # simipolar
    i = np.where(~ np.isnan(rindex[:,0]))[0]
    mod = multi_CNN(smiles[i], rindex[i,0])
    mod.train()
    mod.test()
    
    # nonpolar
    i = np.where(~ np.isnan(rindex[:,1]))[0]
    mod = multi_CNN(smiles[i], rindex[i,1])
    mod.train()
    mod.test()

    # polar
    i = np.where(~ np.isnan(rindex[:,2]))[0]
    mod = multi_CNN(smiles[i], rindex[i,2])
    mod.train()
    mod.test()
    