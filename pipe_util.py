import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
from sklearn.utils.validation import check_is_fitted

from scipy import stats
import cloudpickle

# utility ----------------------------------------------------------------------
# note: need to save/load with clouldpickle for some reason
# pickle or joblib did not work
def write_cp(obj, file_name):
   with open(file_name, 'wb') as f:
    cloudpickle.dump(obj, f)
    return None

def read_cp(file_name):
    with open(file_name, 'rb') as f:
        value = cloudpickle.load(f)
    return value

def find_outlier(s, cutoff = 3):
    assert isinstance(s, pd.Series)
    zs = stats.zscore(s, nan_policy='omit')
    idx = np.where(abs(zs) >= cutoff, True, False)
    return idx

def find_lift(y_true, y_score, n_bins = 10, fn = [len, np.mean, np.median]):
   df = pd.DataFrame({'y_true': y_true, 'y_score': y_score}) 
   dfs = (
        df
        .assign(group = pd.qcut(y_score, q = n_bins))
        .groupby('group')
        .agg(fn)
        .reset_index()
   )
   dfs.columns = dfs.columns.map('_'.join)
   return dfs

# custom transformer and its supporting functions ------------------------------
# Note
# With a newer version of sklearn, OneHotEncoder() will have options like
# OneHotEncoder(max_categories = 3)
# OneHotEncoder(min_frequency = 0.05)
# https://github.com/scikit-learn/scikit-learn/pull/16018
# Accepted on 2022-03-25

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

def keep_level_by_prop(s, prop):
    df_freq = find_prop(s)
    key_list = df_freq.loc[df_freq['prop'] >= prop, 'key'].to_list()
    return key_list

def build_default_dict(s, label):
    def default_key():
        return label
    base_dict = dict(zip(s, s))
    default_dict = defaultdict(default_key, base_dict)
    return default_dict

def build_mapping_by_prop(s, prop, label):
    levels_to_keep = keep_level_by_prop(s, prop)
    d = build_default_dict(levels_to_keep, label)
    return d

class FactorLumpProp(BaseEstimator, TransformerMixin):
    """
    Similar feature as fct_lump_prop() from forcats package in R
    arguments:
        prop = (0, 1] e.g. 0.05. Group factor levels less than 0.01 into one
        label = 'the_other' by default. Label to group
    """
    def __init__(self, prop = 0.05, label = 'the_other'):
        self.prop = prop
        self.label = label
        self.var_list = None

    def fit(self, X, y = None):
        assert isinstance(X, pd.DataFrame)
        var_list = X.columns
        self.columns = var_list
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

    def get_feature_names_out(self, input_features = None):
        return self.var_list

def keep_level_by_top_n(s, top_n):
    df_freq = find_prop(s)
    key_list = df_freq.iloc[:top_n]['key'].to_list()
    return key_list

def build_mapping_by_top_n(s, top_n, label):
    levels_to_keep = keep_level_by_top_n(s, top_n = top_n)
    d = build_default_dict(levels_to_keep, label = label)
    return d


class FactorLumpN(BaseEstimator, TransformerMixin):
    """
    Similar feature as fct_lump_n() from forcats package in R
    arguments:
        top_n = 5 by default. The number of top factors to keep
        label = 'the_other' by default. Label to group
    """
    def __init__(self, top_n = 5, label = 'the_other'):
        self.top_n = top_n
        self.label = label
        self.var_list = None

    def fit(self, X, y = None):
        assert isinstance(X, pd.DataFrame)
        var_list = X.columns
        self.var_list = var_list
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

    def get_feature_names_out(self, input_features = None):
        return self.var_list