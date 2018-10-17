import numpy as np
import abc
from base import TreeQualityPredictor
import pandas as pd
import pickle

class RandomTreeQualityPredictor(TreeQualityPredictor):
    def __init__(self):
        self.Y = None
    
    def predict(self, X):
        predY = np.random.choice(self.Y, X.shape[0])
        if isinstance(X, pd.DataFrame):        
            return pd.Series(predY, index=X.index)
        else:
            return predY
            
    def fit(self, X, Y):
        self.Y = Y
    
    @classmethod
    def load(self, infn):
        with open(infn, "rb") as infd:
            rp = pickle.load(infd)
            return rp
    
    def save(self, outfn):
        with open(outfn, "wb") as outfd:
            pickle.dump(self, outfd)