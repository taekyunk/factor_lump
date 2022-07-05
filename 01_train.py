
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error as mape

# pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

# local modules
# OneHotEncoder() will have max_categories and min_frequency with a newer version
from pipe_util import FactorLumpProp
from pipe_util import FactorLumpN
from pipe_util import read_cp
from pipe_util import write_cp
from pipe_util import find_outlier
from pipe_util import find_prop
from pipe_util import find_lift

# variable importance
from feature_importance import FeatureImportance


# load data --------------------------------------------------------------------
# sample data
import seaborn as sns

df = sns.load_dataset('diamonds')
df
df.info()


# remove outlier ---------------------------------------------------------------

idx = find_outlier(df['price'], cutoff=3)
df = df[~idx]


# split data -------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    df.drop(['price'], axis='columns'),
    df['price'],
    test_size = 0.2,
    random_state = 123
)

x_train
x_train.info()


# build pipeline 


# group features by treatment
var_numeric = ['depth', 'table', 'carat']
var_categorical_group = ['clarity']
var_categorical_binary = ['color']
var_categorical_target = ['cut']

# build one transformer by feature group
numeric_transformer = Pipeline(
    steps = [
        # to impute with mode, use strategy='most_frequent'
        ('simple_imputer', SimpleImputer(strategy = 'mean')), 
        ('scaler', StandardScaler())
    ]
)

categorical_transformer_group = Pipeline(
    steps = [
        ('fct_lump', FactorLumpN(top_n = 3)), 
        ('one_hot_encoder', OneHotEncoder(drop='if_binary'))
    ]
)

categorical_transformer_binary = Pipeline(
    steps = [
        ('as_binary', FunctionTransformer(lambda x: (x == 'G').astype(int))), 
    ]
)
categorical_transformer_target = Pipeline(
    steps = [
        # use recommended settings from 
        # https://github.com/scikit-learn-contrib/category_encoders/issues/327
        ('target_encoder', TargetEncoder(min_samples_leaf=20, smoothing=10))
    ]
)

# combine transformers into ColumnTransformer
preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_transformer, var_numeric),
        ('cat_group', categorical_transformer_group, var_categorical_group),
        ('cat_binary', categorical_transformer_binary, var_categorical_binary),
        ('cat_target', categorical_transformer_target, var_categorical_target),
    ]
)

# Build a final pipeline by modeling step
pipe = Pipeline(
    steps = [
        ('preprocessor', preprocessor),
        ('rf', RandomForestRegressor())
    ]
)

# find parameter names
pipe.get_params().keys()

# note the prefix `rf__` with two _
param_grid = {
    'rf__max_depth': [2, 4, 6],
    'rf__min_samples_split': [2, 3, 4],
    'rf__n_estimators': [100, 120]
}

cv = GridSearchCV(pipe, param_grid, n_jobs = 2)
cv.fit(x_train, y_train)

y_test_pred = cv.predict(x_test)
y_test_pred

# compare the two for overfitting
cv.best_score_
r2_score(y_test, y_test_pred)

mape(y_test, y_test_pred)

# lift
find_lift(y_test, y_test_pred)
# find_lift(y_test, y_test_pred, n_bins = 3)


# select the best model --------------------------------------------------------

# select best model as a pipeline
best_model = cv.best_estimator_
best_model

# variable importance
fi = FeatureImportance(best_model)
fi
fi.get_selected_features()

(fi.get_feature_importance()
    .to_frame('imp')
    .reset_index()
    .sort_values('imp', ascending=False)
)

# save best model
write_cp(best_model, 'best_model.pkl') 
