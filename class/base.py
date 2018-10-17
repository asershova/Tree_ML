import numpy as np
import abc
class TreeQualityPredictor(abc.ABC):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def predict(self, X):
        pass
    
    @abc.abstractmethod
    def fit(self, X, Y):
        pass
    
    @abc.abstractclassmethod
    def load(cls, infn):
        pass
    
    @abc.abstractmethod
    def save(self, outfn):
        pass