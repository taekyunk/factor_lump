
import pandas as pd
import numpy as np

from pipe_util import FactorLumpProp
from pipe_util import FactorLumpN
from pipe_util import find_outlier

# sample data
import seaborn as sns

df = sns.load_dataset('diamonds')

# frequency
df['color'].value_counts()

dfs_clarity = (
    df['clarity']
    .value_counts()
    .to_frame('n')
    .reset_index()
    .assign(prop = lambda x: x['n'] / x['n'].sum())
    )
dfs_clarity


# similar to fct_lump_n() in R
fct_lump_n = FactorLumpN(top_n = 3)
fct_lump_n.fit_transform(df[['color']]).value_counts()

fct_lump_n10 = FactorLumpN(top_n = 10)
fct_lump_n10.fit_transform(df[['color']]).value_counts()

# similar to fct_lump_prop() in R
fct_lump_prop = FactorLumpProp(prop = 0.1)
fct_lump_prop.fit_transform(df[['clarity']]).value_counts()

fct_lump_prop = FactorLumpProp(prop = 0.2)
fct_lump_prop.fit_transform(df[['clarity']]).value_counts()
