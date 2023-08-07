# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
# import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin


class AddColumnGroup():
    def __init__(self,columns,columns_new):
        self.columns=columns
        self.columns_new=columns_new
        
    def transform(self, X):
        X_copy=X.copy()
        for i in range(len(self.columns)):
            for key, value in self.columns_new[i].items():
                X_copy[self.columns[i]].replace(key, value, inplace=True)
        return X_copy
        
    def fit(self, X, y=None):
        return self     


class AddColumnIndex():
    def __init__(self,columns):
        self.columns=columns
    
    def transform(self, X):
        cat_cols = LabelEncoder()
        X_copy=X.copy()
        X_copy[self.columns] = X_copy[self.columns].apply(lambda col:cat_cols.fit_transform(col))
        return X_copy
        
    def fit(self, X, y=None):
        return self

class SMOTETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sampling_strategy=0.1, k_neighbors=7):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors)

    def transform(self, X, y=None):
        return X
         
    def fit(self, X, y):
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        return self
    

class RandomUnderSamplerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sampling_strategy=0.5):
        self.sampling_strategy = sampling_strategy
        self.under_sampler = RandomUnderSampler(sampling_strategy=sampling_strategy)

    def transform(self, X, y=None):
        return X
    
    def fit(self, X, y):
        X_resampled, y_resampled = self.under_sampler.fit_resample(X, y)
        return self