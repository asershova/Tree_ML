#XgboostTreeQualityPredictor
import numpy as np
import abc

import pandas as pd
import pickle
import importlib.machinery
path = '/home/anna/Documents/Mashine_learning/Diploma/materials/Tree_ML/classes/'
loader = importlib.machinery.SourceFileLoader('base', path+'base.py')
base = loader.load_module('base')
from base import TreeQualityPredictor
class XGBoostTreeQualityPredictor(TreeQualityPredictor):
    def __init__(self, xgb):
        self.xgb = xgb
        
    def predict(self, X):
        return self.xgb.predict(X)

    def fit(self, X, Y):
        self.xgb.fit(X, Y)
    
    @classmethod
    def load(cls, infn):
        with open(infn, "rb") as infd:
            xgb = pickle.load(infd)
            return cls(xgb)
    
    def save(self, outfn):
        with open(outfn, "wb") as outfd:
            pickle.dump(self.xgb, outfd)

