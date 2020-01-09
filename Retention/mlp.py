# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:56:07 2020

@author: hcji
"""

import numpy as np
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers
from sklearn.metrics import mean_absolute_error, r2_score


class desc_DNN:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.X_tr, self.X_ts, self.Y_tr, self.Y_ts = train_test_split(X, Y, test_size=0.1)
        
        inp = Input(shape=(X.shape[1:3]))
        n = X.shape[1]
        
        hid = inp
        for i in range(5):
            hid = Dense(32, activation="relu")(hid)
            n = int(n * 0.5)
        
        prd = Dense(1, activation="linear")(hid)
        opt = optimizers.Adam(lr=0.001)
        model = Model(inp, prd)
        model.compile(optimizer=opt, loss='mse', metrics=['mae'])
        self.model = model
    
    def train(self, epochs=50):
        self.model.fit(self.X_tr, self.Y_tr, epochs=epochs)
    
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
    
    def clear(self):
        K.clear_session()
    
    def save(self, path):
        self.model.save(path)
        K.clear_session()


if __name__ == '__main__':
    
    
    import json
    from scipy.sparse import load_npz
    
    with open('DeepEI/data/split.json', 'r') as js:
        keep = np.array(json.load(js)['keep'])
    descriptors = np.load('DeepEI/data/descriptors.npy')[keep,:]
    morgan = load_npz('DeepEI/data/morgan.npz')[keep,:].todense()
    rindex = np.load('DeepEI/data/retention.npy')[keep,:]
    
    # descriptor
    ## simipolar
    i = np.where(~ np.isnan(rindex[:,0]))[0]
    mod = desc_DNN(descriptors[i], rindex[i,0])
    mod.train()
    mod.test()
    mod.clear()
    
    ## nonpolar
    i = np.where(~ np.isnan(rindex[:,1]))[0]
    mod = desc_DNN(descriptors[i], rindex[i,0])
    mod.train()
    mod.test()
    mod.clear()

    ## polar
    i = np.where(~ np.isnan(rindex[:,2]))[0]
    mod = desc_DNN(descriptors[i], rindex[i,0])
    mod.train()
    mod.test()
    mod.clear()
    
    # morgan
    ## simipolar
    i = np.where(~ np.isnan(rindex[:,0]))[0]
    mod = desc_DNN(morgan[i], rindex[i,0])
    mod.train()
    mod.test()
    mod.clear()
    
    ## nonpolar
    i = np.where(~ np.isnan(rindex[:,1]))[0]
    mod = desc_DNN(morgan[i], rindex[i,0])
    mod.train()
    mod.test()
    mod.clear()

    ## polar
    i = np.where(~ np.isnan(rindex[:,2]))[0]
    mod = desc_DNN(morgan[i], rindex[i,0])
    mod.train()
    mod.test()
    mod.clear()    