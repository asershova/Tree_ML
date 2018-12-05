from sys import argv
from base import TreeQualityPredictor
import pandas as pd
import pickle
import lightgbm as lgb
from lgb_opt_tqp import LightGBMTreeQualityPredictor

seed = [248,
        655,
        219,
        329,
        389,
        180,
        901,
        676,
        217,
        23]

for n in [15, 20, 25, 30]:
    for i in range(10):
        tqp3 = LightGBMTreeQualityPredictor(seed=seed[i])
        
        train_all = pd.read_csv('../data/{}_train.txt'.format(n), sep = '\t', index_col = 0)
        X = train_all.iloc[:,:-3]
        y = train_all.iloc[:,-2]
        
        tqp3.fit(X, y)
        tqp3.save('../models/lgb_opt_{}_r{}.txt'.format(n, i))
