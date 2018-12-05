#LightGBMTreeQualityPredictor
import numpy as np
import abc
from base import TreeQualityPredictor
import pandas as pd
import pickle
import lightgbm as lgb


class LightGBMTreeQualityPredictor(TreeQualityPredictor):
    def __init__(self, seed=None):
        self.gbm = lgb.LGBMRegressor
        self.params = {'learning_rate': 0.05,
                     'max_depth': 2,
                     'n_estimators': 200,
                     'n_jobs': -1,
                     'random_state': seed}
        
    def predict(self, X):
        return self.gbm.predict(X)

    def fit(self, X, Y):
        lgb_data = lgb.Dataset(X, Y)
        self.gbm = lgb.train(self.params, lgb_data)
    
    @classmethod
    def load(cls, infn):
        with open(infn, "rb") as infd:
            gbm = pickle.load(infd)
            return cls(gbm)
    
    def save(self, outfn):
        with open(outfn, "wb") as outfd:
            pickle.dump(self.gbm, outfd)
