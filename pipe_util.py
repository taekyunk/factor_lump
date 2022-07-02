
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
from sklearn.utils.validation import check_is_fitted


def find_outlier(s, cutoff = 2):
    assert isinstance(s, pd.Series)
    zs = stats.zscore(s, nan_policy='omit')
    idx = np.where(abs(zs) >= cutoff, True, False)
    return idx


def find_prop(s):
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    dfs = (
        s
        .value_counts()
        .rename_axis(index = 'key')
        .to_frame('n')
        .reset_index()
        .sort_values('n', ascending=False)
        .assign(prop = lambda x: x['n']/ x['n'].sum())
    )
    return dfs

def keep_level_by_prop(s, prop = 0.01):
    df_freq = find_prop(s)
    key_list = df_freq.loc[df_freq['prop'] >= prop, 'key'].to_list()
    return key_list

def build_default_dict(s, label = 'zzz'):
    def default_key():
        return label
    base_dict = dict(zip(s, s))
    default_dict = defaultdict(default_key, base_dict)
    return default_dict

def build_mapping_by_prop(s, prop = 0.01, label = 'zzz'):
    levels_to_keep = keep_level_by_prop(s, prop)
    d = build_default_dict(levels_to_keep, label)
    return d

class FactorLumpProp(BaseEstimator, TransformerMixin):
   # initializer 
    def __init__(self, prop, label = 'zzz'):
        self.prop = prop
        self.label = label

    def fit(self, X, y = None):
        assert isinstance(X, pd.DataFrame)
        var_list = X.columns
        mapping_dict = dict()
        for var in var_list:
            d = build_mapping_by_prop(X[var], prop = self.prop, label = self.label)
            mapping_dict[var] = d
        self.mapping_dict = mapping_dict
        return self
    
    def transform(self, X, y = None):
        assert isinstance(X, pd.DataFrame)
        check_is_fitted(self, ['mapping_dict'])

        X = X.copy()
        var_list = X.columns
        for var in var_list:
            d = self.mapping_dict[var]
            X[var] = X[var].map(d)
        return X

def keep_level_by_top_n(s, top_n = 5):
    df_freq = find_prop(s)
    key_list = df_freq.iloc[:top_n]['key'].to_list()
    return key_list

def build_mapping_by_top_n(s, top_n = 5, label = 'zzz'):
    levels_to_keep = keep_level_by_top_n(s, top_n = top_n)
    d = build_default_dict(levels_to_keep, label = label)
    return d


class FactorLumpN(BaseEstimator, TransformerMixin):
   # initializer 
    def __init__(self, top_n = 5, label = 'zzz'):
        self.top_n = top_n
        self.label = label

    def fit(self, X, y = None):
        assert isinstance(X, pd.DataFrame)
        var_list = X.columns
        mapping_dict = dict()
        for var in var_list:
            d = build_mapping_by_top_n(X[var], top_n = self.top_n, label = self.label)
            mapping_dict[var] = d
        self.mapping_dict = mapping_dict
        return self
    
    def transform(self, X, y = None):
        assert isinstance(X, pd.DataFrame)
        check_is_fitted(self, ['mapping_dict'])

        X = X.copy()
        var_list = X.columns
        for var in var_list:
            d = self.mapping_dict[var]
            X[var] = X[var].map(d)
        return X
