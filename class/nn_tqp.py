#NearestNeighbourTreeQualityPredictor
import numpy as np
import abc

import pandas as pd
import pickle
import importlib.machinery
path = '/home/anna/Documents/Mashine_learning/Diploma/materials/Tree_ML/classes/'
loader = importlib.machinery.SourceFileLoader('base', path+'base.py')
base = loader.load_module('base')
from base import TreeQualityPredictor
class NNTreeQualityPredictor(TreeQualityPredictor):
    def __init__(self, nn):
        self.nn = nn
        
    def predict(self, X):
        return self.nn.predict(X)

    def fit(self, X, Y):
        self.nn.fit(X, Y)
    
    @classmethod
    def load(cls, infn):
        with open(infn, "rb") as infd:
            nn = pickle.load(infd)
            return cls(nn)
    
    def save(self, outfn):
        with open(outfn, "wb") as outfd:
            pickle.dump(self.nn, outfd)
