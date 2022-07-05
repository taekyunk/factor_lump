
# About

Custom transformer to work with categorical variables

- `FactorLumpProp(prop = 0.05)`: Similar to `fct_lump_prop()` in R
- `FactorLumpN(top_n=5)`: Similar to `fct_lump_n()` in R

Other utility functions
- `read_cp()`: read object using cloudpickle
- `write_cp()`: write object using cloudpickle
- `find_lift()`: find lift and returns a dataframe
- `find_prop()`: find the frequency and probability for a `pd.Series`

Class `FeatureImportance()` adapted from 
- [notebook](https://www.kaggle.com/code/kylegilde/extracting-scikit-feature-names-importances/notebook)
- [github discussion](https://github.com/scikit-learn/scikit-learn/issues/12525#issuecomment-1071203398)

# Note

- Soon, `OneHotEncoder()` will gain options to collapse infrequent factor levels
    - [Source](https://github.com/scikit-learn/scikit-learn/pull/16018)
- This code is a temporary solution when the new sklearn is not available

# Author

- Taekyun (TK) Kim: taekyunk@gmail.com

