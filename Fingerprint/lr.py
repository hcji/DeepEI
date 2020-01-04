# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:19:57 2020

@author: hcji
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class LR:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.X_tr, self.X_ts, self.Y_tr, self.Y_ts = train_test_split(X, Y, test_size=0.1)
    
    def train(self):
        self.model = LogisticRegression().fit(self.X_tr, self.Y_tr)
    
    def test(self):
        Y_pred = np.round(self.model.predict(self.X_ts))
        f1 = f1_score(self.Y_ts, Y_pred)
        precision = precision_score(self.Y_ts, Y_pred)
        recall = recall_score(self.Y_ts, Y_pred)
        accuracy = accuracy_score(self.Y_ts, Y_pred)
        return accuracy, precision, recall, f1