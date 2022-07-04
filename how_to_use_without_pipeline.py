
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from pipe_util import FactorLumpProp
from pipe_util import FactorLumpN
from pipe_util import find_outlier
from pipe_util import find_prop

# sample data
import seaborn as sns
df = sns.load_dataset('diamonds')


# with one data set only (without train / test) --------------------------------
# similar to fct_lump_n() in R
fct_lump_n = FactorLumpN(top_n = 3)
# note that the argument requires a *dataframe*, not a series
dfc1 = fct_lump_n.fit_transform(df[['color']])
find_prop(dfc1['color'])
find_prop(df['color'])


# if the top_n is bigger than the available levels, returns unmodified factor
fct_lump_n10 = FactorLumpN(top_n = 10)
dfc2 = fct_lump_n10.fit_transform(df[['color']])
find_prop(dfc2['color'])
find_prop(df['color'])


# similar to fct_lump_prop() in R
fct_lump_prop = FactorLumpProp(prop = 0.1)
dfc3 = fct_lump_prop.fit_transform(df[['clarity']])
find_prop(dfc3['clarity'])
find_prop(df['clarity'])


fct_lump_prop = FactorLumpProp(prop = 0.2)
dfc4 = fct_lump_prop.fit_transform(df[['clarity']])
find_prop(dfc4['clarity'])
find_prop(df['clarity'])


# with split data --------------------------------------------------------------

numeric_features = ['depth', 'table', 'carat']
categorical_features = ['color', 'clarity']
# note that this is *not* a list
dependent_variable = 'price'

x_train, x_test, y_train, y_test = train_test_split(
    df[numeric_features + categorical_features],
    df[dependent_variable],
    test_size = 0.2,
    random_state = 123
)

fct_lump_n = FactorLumpN(top_n = 3)
# fit with training
fct_lump_n.fit(x_train[categorical_features])
fct_lump_n.transform(x_train[categorical_features])
# transform with test
fct_lump_n.transform(x_test[categorical_features])

fct_lump_prop = FactorLumpProp(prop = 0.1)
# fit with training
fct_lump_prop.fit(x_train[categorical_features])
fct_lump_prop.transform(x_train[categorical_features])
# transform with test
fct_lump_prop.transform(x_test[categorical_features])
