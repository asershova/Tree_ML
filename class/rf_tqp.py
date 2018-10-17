#RandomForestTreeQualityPredictor
import numpy as np
import abc
from base import TreeQualityPredictor
import pandas as pd
import pickle


class RFTreeQualityPredictor(TreeQualityPredictor):
    def __init__(self, rf):
        self.rf = rf
        
    def predict(self, X):
        return self.rf.predict(X)

    def fit(self, X, Y):
        self.rf.fit(X, Y)
    
    @classmethod
    def load(cls, infn):
        with open(infn, "rb") as infd:
            rf = pickle.load(infd)
            return cls(rf)
    
    def save(self, outfn):
        with open(outfn, "wb") as outfd:
            pickle.dump(self.rf, outfd)
