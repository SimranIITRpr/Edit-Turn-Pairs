#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:35:14 2020

@author: simransetia
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
from sklearn.svm import SVC 
from sklearn import metrics
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2,f_classif
from sklearn.metrics import classification_report, confusion_matrix 
data = pd.read_csv("/Users/simransetia/Documents/simran/Laboratory/talk page analysis/features1")
header=None
data=np.array(data)

X = data[:,1:659]
y = data[:,659]
y=y.astype('int')


from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
pca = PCA(n_components=100)
Xkbest= pca.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(Xkbest, y, test_size = 0.20) 
#Hyperparameter tuning
'''
svclassifier = SVC(kernel='rbf')  
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test) 
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#scores = cross_val_score(svclassifier, X, y, cv=5, scoring='f1_micro')

from sklearn.model_selection import GridSearchCV 
  
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(svclassifier, param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_train, y_train)
# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 
grid_predictions = grid.predict(X_test) 
  
# print classification report 
print(classification_report(y_test, grid_predictions))
'''
svclassifier = SVC(kernel='rbf',C=1,gamma=0.1) #Hyperparamters tuned for ETP gold v-1.0
#svclassifier = SVC(kernel='rbf',C=10,gamma=0.1) #Hyperparamters tuned for ETP gold v-2.0
scores = cross_val_score(svclassifier, Xkbest, y, cv=5, scoring='f1_micro')
svclassifier.fit(X_train, y_train)

cv = KFold(n_splits=5)
y_pred = cross_val_predict(svclassifier, X_test, y_test, cv = cv)
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))
