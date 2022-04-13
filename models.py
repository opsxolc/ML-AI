import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import datasets, linear_model, metrics, tree
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier

import os
import warnings

warnings.filterwarnings('ignore')

dataset = pd.read_csv('winequality-red.csv', delimiter=',')
dataset_X = dataset.iloc[:, :-1]
dataset_y = dataset['quality']

X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y, random_state=1488)

#----------------------------

ovo_ridge = OneVsOneClassifier(linear_model.RidgeClassifier(random_state=1488))
ovo_ridge.fit(X_train, y_train)
y_pred = ovo_ridge.predict(X_test)

print("OneVsOne Ridge Classifier")
print("Accuracy score:", metrics.accuracy_score(y_test, y_pred))
print("Cross validation mean:", np.mean(cross_val_score(ovo_ridge, dataset_X, dataset_y, scoring = 'accuracy', cv = 7)))

#----------------------------

ovr_ridge = OneVsRestClassifier(linear_model.RidgeClassifier(random_state=1488))
ovr_ridge.fit(X_train, y_train)
y_pred = ovr_ridge.predict(X_test)

print("OneVsRest Ridge Classifier")
print("Accuracy score:", metrics.accuracy_score(y_test, y_pred))
print("Cross validation mean:", np.mean(cross_val_score(ovr_ridge, dataset_X, dataset_y, scoring = 'accuracy', cv = 7)))

#----------------------------

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)

print("Decision Tree Classifier")
print("Accuracy score:", metrics.accuracy_score(y_test, y_pred))
print("Cross validation mean:", np.mean(cross_val_score(dtree, dataset_X, dataset_y, scoring = 'accuracy', cv = 7)))

#----------------------------

xgb_cl = xgb.XGBClassifier()
xgb_cl.fit(X_train, y_train)
y_pred = xgb_cl.predict(X_test)

print("XGBoost Classifier")
print("Accuracy score:", metrics.accuracy_score(y_test, y_pred))
print("Cross validation mean:", np.mean(cross_val_score(xgb_cl, dataset_X, dataset_y, scoring = 'accuracy', cv = 7)))
