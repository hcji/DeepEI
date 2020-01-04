# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:21:27 2019

@author: hcji
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plsda = pd.read_csv('Result/fingerprint_PLSDA.csv')
fcnn = pd.read_csv('Result/fingerprint_DNN.csv')
plt.plot(plsda['frac'], plsda['accuracy'], '.', label='PLSDA')
plt.plot(fcnn['frac'], fcnn['accuracy'], '.', color = 'red', label='FCNN')
plt.ylabel('Accuracy')
plt.xlabel('Positive Fraction')
plt.legend(['PLSDA', 'FCNN'], loc='lower right')


rank_nist = pd.read_csv('Result/rank_nist.csv')
candidate_nist = pd.read_csv('Result/candidates_nist.csv')
plt.hist(rank_nist['fp_score'])
plt.ylabel('Compounds')
plt.xlabel('Jaccard Score')