
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
from pipe_util import DropHighlyCorrelated
from pipe_util import read_cp
from pipe_util import write_cp
from pipe_util import find_outlier
from pipe_util import find_prop
from pipe_util import find_lift

# variable importance
# from feature_names import get_feature_names
from feature_names2 import get_feature_names
from feature_importance import FeatureImportance

from plotnine import ggplot, aes, geom_point, geom_line, facet_wrap

# load data --------------------------------------------------------------------
# sample data from seaborn
# import seaborn as sns

# df = sns.load_dataset('diamonds')
# df.to_csv('diamonds2.csv')
df = pd.read_csv('diamonds.csv')
df
df.info()


# remove outlier ---------------------------------------------------------------

idx = find_outlier(df['price'], cutoff=3)
df = df[~idx]

# drop correlated --------------------------------------------------------------

drop_correlated = DropHighlyCorrelated(
    threshold=0.85, 
    # select numeric variables here
    candidate = ['depth', 'table', 'carat', 'x', 'y', 'z'])
df = drop_correlated.fit_transform(df)
df
# dropped features
drop_correlated.correlated_features

# split data -------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    df.drop(['price'], axis='columns'),
    df['price'],
    test_size = 0.2,
    random_state = 123
)

x_train
x_train.info()


# build pipeline ---------------------------------------------------------------

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

# fit preprocessor only --------------------------------------------------------
# fit processor separately to reduce time
# put this in a pipeline to use FeatureImportance
pp = preprocessor
x_train_mat = pp.fit_transform(x_train, y_train)

get_feature_names(pp)


# Build a final pipeline by modeling step
pipe = Pipeline(
    steps = [
        ('preprocessor', preprocessor),
    ]
)
pipe.fit(x_train, y_train)

fi = FeatureImportance(pipe)
fi.get_feature_names()

# CV ---------------------------------------------------------------------------

param_grid = {
    'max_features': ['sqrt', 'log2'],
    'n_estimators': [100, 120, 140],
}

cv = GridSearchCV(RandomForestRegressor(), param_grid, n_jobs = 2, verbose=2)
cv.fit(x_train_mat, y_train)

# select the best model --------------------------------------------------------

x_test_mat = preprocessor.transform(x_test)
y_test_pred = cv.predict(x_test_mat)
y_test_pred

# compare the two for overfitting
cv.best_score_
r2_score(y_test, y_test_pred)

mape(y_test, y_test_pred)

# lift
dfs = find_lift(y_test, y_test_pred)
dfs['idx'] = range(dfs.shape[0])
dfs

dfst = dfs.melt(id_vars = ['idx'], value_vars = ['y_true_mean', 'y_score_mean'])
dfst

(ggplot(dfst, aes('idx', 'value', color = 'variable'))
+ geom_point()
+ geom_line()
)

(ggplot(dfst, aes('idx', 'value', color = 'variable'))
+ geom_point()
+ geom_line()
+ facet_wrap('~ variable')
)

# variable importance

dfi = (
    pd.DataFrame({
        'name': get_feature_names(pp),
        'importance': cv.best_estimator_.feature_importances_
    })
    .sort_values('importance', ascending=False)
)
dfi

# find the best model ----------------------------------------------------------

# save only the parameters
best_param = cv.best_params_
write_cp(best_param, 'best_param.pkl')

