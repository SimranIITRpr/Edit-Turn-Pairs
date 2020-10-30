#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:10:25 2020

@author: simransetia
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
from sklearn import metrics
from sklearn.feature_selection import SelectKBest 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import GridSearchCV 
data = pd.read_csv("/Users/simransetia/Documents/simran/Laboratory/talk page analysis/features2.csv")
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
clf = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(Xkbest, y, test_size = 0.20) 
'''

clf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=15, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=800,
                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)   #ETP gold v-1.0
'''
clf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=25, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=10,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)  #ETP gold v-2.0
clf.fit(Xkbest, y)
y_pred =clf.predict(X_test) 
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
scores = cross_val_score(clf, Xkbest, y, cv=10, scoring='f1_micro')
'''
n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 
hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

#gridF = GridSearchCV(clf, hyperF, cv = 5, verbose = 1, n_jobs = -1)
gridF = GridSearchCV(clf, hyperF, refit = True, verbose = 3) 
bestF = gridF.fit(X_train, y_train)
'''