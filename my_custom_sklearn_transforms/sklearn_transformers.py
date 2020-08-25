from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample
import pandas as pd
import numpy as np

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
    
# All sklearn Transforms must have the `transform` and `fit` methods
class BalancedDataset(BaseEstimator, TransformerMixin):
    def __init__(self, minority):
        self.minority = minority
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        data_minority = resample(self.minority, 
                                 replace=True,
                                 n_samples=8000,
                                 random_state=123)
        data_upsampled = np.concatenate([data, data_minority.values], axis=0)
        return data_upsampled
