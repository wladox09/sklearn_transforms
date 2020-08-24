from sklearn.base import BaseEstimator, TransformerMixin


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
class SimpleImputerMean(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Primero copiamos el dataframe de entrada 'X' de entrada
        data = X.copy()
        data_target = data_copy.OBJETIVO
        data_predictors = data.drop(['OBJETIVO'], axis=1)
        # predictores numéricos.. 
        data_numeric_predictors = data_predictors.select_dtypes(exclude=['object'])
        # Imputation
        my_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        data_final = pd.DataFrame(my_imputer.fit_transform(data_numeric_predictors))
        data_final.columns = data_predictors.columns
        # Reintegrando el target
        data_final['OBJETIVO'] = data_target
        return data_final
