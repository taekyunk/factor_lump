
import pandas as pd
import numpy as np

# custom util
from pipe_util import DropHighlyCorrelated
from feature_importance import FeatureImportance


# load data --------------------------------------------------------------------
# sample data
import seaborn as sns

df = sns.load_dataset('diamonds')
df
df.info()


# drop highly correlatd --------------------------------------------------------

# select numeric variables
var_numeric = ['depth', 'table', 'carat', 'x', 'y', 'z']

# drop highly correlated numeric
hc = DropHighlyCorrelated(threshold = 0.8)
dfn = hc.fit_transform(df[var_numeric])
hc.get_feature_names_out()

# combine back with categorical
dfc = df.drop(var_numeric, axis='columns')
dfc

df_new = pd.concat([dfc, dfn], axis = 'columns')
df_new
