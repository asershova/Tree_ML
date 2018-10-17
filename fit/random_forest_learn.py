#!/usr/bin/python
# Trainer of RandomForestClassifier. Input - file with training data, n_estimators, output - trained model (pickle object)
#random_forest_learn.py ../data/15_train.txt 200 ../models/rf_200_15.txt 
from sys import argv
from base import TreeQualityPredictor
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from rf_tqp import RFTreeQualityPredictor

rf = RandomForestRegressor(n_estimators=int(argv[2]))
tqp = RFTreeQualityPredictor(rf)

train_all = pd.read_csv(argv[1], sep = '\t', index_col = 0)
X = train_all.iloc[:,:-3]
y = train_all.iloc[:,-2]

tqp.fit(X, y)
tqp.save(argv[3])
