
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from pipe_util import FactorGroup
from pipe_util import find_outlier

# sample data
import seaborn as sns
sns.get_dataset_names()

df = sns.load_dataset('diamonds')
df
df.columns

numeric_features = ['depth', 'table', 'carat']
categorical_features = ['color', 'clarity']
# note that this is *not* a list
dependent_variable = 'price'


# split data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    df[numeric_features + categorical_features],
    df[dependent_variable],
    test_size = 0.2,
    random_state = 123
)

x_train

# 1. remove outlier from training only

idx = find_outlier(y_train)
idx

x_train1 = x_train[~idx]
y_train1 = y_train[~idx]


# outlier = OutlierRemover('price')
# outlier.fit(dfe)
# dfef = outlier.transform(dfe)
# (dfe.shape, dfef.shape)


# 2. rest of the pipeline

numeric_transformer = Pipeline(
    steps = [
        ('simple_imputer', SimpleImputer(strategy = 'mean')), 
        ('scaler', StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps = [
        ('fct_lump', FactorGroup(prop = 0.1)), 
        ('one_hot_encoder', OneHotEncoder())
    ]
)

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipe = Pipeline(
    steps = [
        ('preprocessor', preprocessor)
    ]
)

# split the data here

