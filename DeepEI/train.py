# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 07:47:48 2019

@author: hcji
"""

import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

'''
train a model to predict retention index
'''
rindex = np.load('Data/RI_data.npy')
morgan = np.load('Data/Morgan_fp.npy')

# check the number of data
n_SimiStdNP = len(np.where(~ np.isnan(rindex[:,0]))[0])
n_StdNP = len(np.where(~ np.isnan(rindex[:,1]))[0])
n_StdPolar = len(np.where(~ np.isnan(rindex[:,2]))[0])

def build_RI_model(morgan, RI):
    # remove nan
    keep = np.where(~ np.isnan(RI))[0]
    X = morgan[keep,:]
    Y = RI[keep]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    
    # train model
    layer_in = Input(shape=(X.shape[1],), name="morgan_fp")
    layer_dense = layer_in
    n_nodes = X.shape[1]
    for j in range(5):
        layer_dense = Dense(int(n_nodes), activation="relu")(layer_dense)
        n_nodes = n_nodes * 0.5
    layer_output = Dense(2, activation="linear", name="output")(layer_dense)
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
    Y_predict = model.predict(X_test)[:,0]
    correlation = round(pearsonr(Y_predict, Y_test)[0], 4)
    plt.cla()
    plt.plot(Y_test, Y_predict, '.', color = 'blue')
    plt.plot([0,4500], [0,4500], color ='red')
    plt.ylabel('Predicted RI')
    plt.xlabel('Experimental RI')        
    plt.text(0, 4000, 'R2='+str(correlation), fontsize=15)
    plt.show()
    


'''
train models to predict fingerprint
'''
spec = np.load('Data/Peak_data.npy')
cdk_fp = np.load('Data/CDK_fp.npy')

def build_FP_model(morgan, FP):
    # check bias
    

