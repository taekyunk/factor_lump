
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

#-------------------------------------------------------------------------------

# get some data set to score
import seaborn as sns
df = sns.load_dataset('diamonds')
df

# randomly select some data 
# assume that there is a id field
df_new = (
    df
    .assign(id = range(df.shape[0]))
    .sample(n=200, random_state=123)
    .reset_index()
)
df_new
df_new.info()


# load model
model = read_cp('best_model.pkl')

# score
y_pred = model.predict(df_new)
y_pred


# combine with id
df_pred = pd.DataFrame({'id': df_new['id'], 'pred': y_pred})
df_pred

# write to table
