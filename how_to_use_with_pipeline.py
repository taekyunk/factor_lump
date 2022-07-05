
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error as mape

from pipe_util import FactorLumpProp
from pipe_util import FactorLumpN
from pipe_util import find_outlier
from pipe_util import read_cp
from pipe_util import write_cp
from pipe_util import find_prop

from feature_importance import FeatureImportance

#-------------------------------------------------------------------------------
# sample data
import seaborn as sns

df = sns.load_dataset('diamonds')
df
df.columns

df['color'].value_counts()
df['clarity'].value_counts()

#-------------------------------------------------------------------------------
# group features by treatment

numeric_features = ['depth', 'table', 'carat']
categorical_features1 = ['color']
categorical_features2 = ['clarity']
categorical_features = categorical_features1 + categorical_features2
# note that this is *not* a list
dependent_variable = 'price'

# test changing field 

# this does not affect pipeline 
# df['depth'] = df['depth'].astype(str)

# split data

x_train, x_test, y_train, y_test = train_test_split(
    df[numeric_features + categorical_features],
    df[dependent_variable],
    test_size = 0.2,
    random_state = 123
)

x_train
x_train.info()

# 1. remove outlier from training only

idx = find_outlier(y_train)
idx

x_train1 = x_train[~idx]
y_train1 = y_train[~idx]



# 2. build transformers for each variable group that gets different treatment

numeric_transformer = Pipeline(
    steps = [
        # to impute with mode, use strategy='most_frequent'
        ('simple_imputer', SimpleImputer(strategy = 'mean')), 
        ('scaler', StandardScaler())
    ]
)

categorical_transformer1= Pipeline(
    steps = [
        ('fct_lump', FactorLumpProp(prop = 0.1)), 
        ('one_hot_encoder', OneHotEncoder())
    ]
)

categorical_transformer2 = Pipeline(
    steps = [
        ('fct_lump', FactorLumpN(top_n = 3)), 
        ('one_hot_encoder', OneHotEncoder())
    ]
)

# combine each transfomers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat1', categorical_transformer1, categorical_features1),
        ('cat2', categorical_transformer2, categorical_features2)
    ]
)

pipe = Pipeline(
    steps = [
        ('preprocessor', preprocessor)
    ]
)

# split the data here

pipe.fit(x_train)

x_train_updated = pipe.transform(x_train)
x_train_updated
x_test_updated = pipe.transform(x_test)
x_test_updated

# with sklearn 1.1 and python 3.8, this works
# pipe.get_feature_names_out()

# only one fit ------------------------------


pipe_rf = Pipeline(
    steps = [
        ('preprocessor', preprocessor),
        ('rf', RandomForestRegressor())
    ]
)

pipe_rf.fit(x_train, y_train)

pipe_rf.get_params()
pipe_rf.named_steps
pipe_rf.predict(x_test)

fi = FeatureImportance(pipe_rf)
fi
# name is sort of incorrect but at least level works
fi.get_feature_importance()
fi.get_feature_names()

# save
write_cp(pipe_rf, 'rf1.pkl')
new_pipe = read_cp('rf1.pkl')
new_pipe.predict(x_test)


# GridsearchCV() with pipeline -------------------------------------------------


pipe_rf = Pipeline(
    steps = [
        ('preprocessor', preprocessor),
        ('rf', RandomForestRegressor())
    ]
)

# find parameter names
pipe_rf.get_params().keys()

# note the prefix `rf__`
param_grid = {
    'rf__max_depth': [2, 4, 6],
    'rf__min_samples_split': [2, 3, 4],
    'rf__n_estimators': [100, 120]
}


crf = GridSearchCV(pipe_rf, param_grid, n_jobs = 2)
crf.fit(x_train, y_train)

crf.best_score_
crf.best_params_
crf.cv_results_

y_test_pred = crf.predict(x_test)
y_test_pred



best_model = crf.best_estimator_
best_model
str(best_model)
best_model.fit(x_train, y_train)
y_test_pred = best_model.predict(x_test)
y_test_pred


# model comparison with metrics
# - mape, r2
r2_score(y_test, y_test_pred)
mape(y_test, y_test_pred)



# variable importance
# need to extract a pipeline, not a GridSearchCV object
fi = FeatureImportance(crf.best_estimator_, verbose=True)
fi
fi.get_selected_features()
fi.get_feature_importance()
# need plotly for this
# feature_importance.plot(top_n_features=25)

# to do





# save and load pipeline
write_cp(best_model, 'best_model.pkl')
new_model = read_cp('best_model.pkl')
type(new_model)
new_model.predict(x_test)



#

# best_model.get_feature_names_out()
best_model[:-1].get_feature_names_out()

# pipe.get_feature_names_out()
# pipe[:-1].get_feature_names_out()

# how to investigate pipeline object
pipe.named_steps
type(pipe.named_steps)
pipe.named_steps['preprocessor']

# this not helpful
pipe.feature_names_in_
crf.feature_names_in_

