class AddColumnGroup():
    def __init__(self,columns,columns_new):
        self.columns=columns
        self.columns_new=columns_new
        
    def transform(self, X):
        X_copy=X.copy()
        for i in range(self.columns):
            for key, value in self.columns_new[i]:
                X_copy[self.columns[i]].replace(key, value)
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