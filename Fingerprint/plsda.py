# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:03:40 2020

@author: hcji
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

class PLSDA:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.X_tr, self.X_ts, self.Y_tr, self.Y_ts = train_test_split(X, Y, test_size=0.1)
        self.X_tr, self.X_dev, self.Y_tr, self.Y_dev = train_test_split(self.X_tr, self.Y_tr, test_size=0.11)
        
    def train(self, ncomps=range(2,9)):
        self.model = None
        best_acc = -1
        Y_dev = np.round(self.Y_dev[:,0])
        for n in ncomps:
            plsda = PLSRegression(n_components=n).fit(X=self.X_tr, Y=self.Y_tr)
            Y_valhat = plsda.predict(self.X_dev)
            Y_valhat = np.round(Y_valhat[:,0])
            Y_val_acc = accuracy_score(Y_dev, Y_valhat)
            if Y_val_acc > best_acc:
                best_acc = Y_val_acc
                self.model = plsda
    
    def test(self):
        Y_pred = np.round(self.model.predict(self.X_ts))
        f1 = f1_score(self.Y_ts[:,0], Y_pred[:,0])
        precision = precision_score(self.Y_ts[:,0], Y_pred[:,0])
        recall = recall_score(self.Y_ts[:,0], Y_pred[:,0])
        accuracy = accuracy_score(self.Y_ts[:,0], Y_pred[:,0])
        return accuracy, precision, recall, f1
        