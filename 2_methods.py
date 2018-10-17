# Command-line arguments: method1, method2 (pickle objects of fitted predictors), test data

from sys import argv
import numpy as np
import abc
import random
import pandas as pd
from sklearn import metrics
import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from random_rfd import RandomTreeQualityPredictor
from rf_tqp import RFTreeQualityPredictor

def prox (y_pred_1, y_pred_2, y_test):
    count_1 = 0
    count_2 = 0
    for i in range(len(y_test)):
        if abs(y_pred_1[i] - y_test[i]) > abs(y_pred_2[i] - y_test[i]):
            count_2+=1
        if abs(y_pred_1[i] - y_test[i]) < abs(y_pred_2[i] - y_test[i]):    
            count_1+=1
    return (count_1, count_2)

def mean_error(y_pred, y_test):
    err = 0
    for i in range(len(y_test)):
        dif = y_pred[i] - y_test[i]
        err = err+dif
    return (err/len(y_test))


test = pd.read_csv(argv[3], sep = '\t', index_col = 0)
X_test = test.iloc[:,:-3]
y_test = test.iloc[:,-2]
with open(argv[1], 'rb') as f1:
    m1 = pickle.load(f1)
with open(argv[2], 'rb') as f2:
    m2 = pickle.load(f2)

y_pred_1 = m1.predict(X_test)
y_pred_2 = m2.predict(X_test)

mse_1 = metrics.mean_squared_error(y_test, y_pred_1)
mse_2 = metrics.mean_squared_error(y_test, y_pred_2)

#mae_1 = metrics.mean_absolute_error(y_test, y_pred_1)
#mae_2 = metrics.mean_absolute_error(y_test, y_pred_2)
me_1 = mean_error(y_test, y_pred_1)
me_2 = mean_error(y_test, y_pred_2)

a,b = prox(y_pred_1, y_pred_2, y_test)

print ("First method MSE: ", mse_1,
       "\nFirst method ME: ", me_1,
       "\nSecond method MSE: ", mse_2,
       "\nSecond method ME: ", me_2,
       "\nFirst method is the best: %.2f" % a,
       "\nSecond method is the best: %.2f" % b)
