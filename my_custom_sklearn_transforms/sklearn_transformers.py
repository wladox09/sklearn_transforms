from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample

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
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Separate majority and minority classes
        data_majority = data[data.OBJETIVO=='Aceptado']
        data_minority = data[data.OBJETIVO=='Sospechoso']
        # Upsample minority class
        data_minority_upsampled = resample(data_minority, 
                                         replace=True,
                                         n_samples=8000,
                                         random_state=123)
        # Combine majority class with upsampled minority class
        data_upsampled = pd.concat([data_majority, data_minority_upsampled])
        data_upsampled = pd.concat([data_upsampled, data_minority])
        return data_upsampled
