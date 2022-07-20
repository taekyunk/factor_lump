
# based on the code from Kyle Glide
# https://www.kaggle.com/code/kylegilde/extracting-scikit-feature-names-importances/notebook
#
# updated
# - add a condition to check get_feature_names_out() first
#    https://github.com/scikit-learn/scikit-learn/issues/12525#issuecomment-1071203398
# - Modified to remove the dependency on plotly 
# - exract only one function
# by Taekyun Kim (taekyunk@gmail.com)

import numpy as np  
import pandas as pd  
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


def get_feature_names(column_transformer):
    """
    Get feature names from a fitted column transformer
    """
    assert isinstance(column_transformer, ColumnTransformer), "Input isn't a ColumnTransformer"
    check_is_fitted(column_transformer)

    new_feature_names = []

    for i, transformer_item in enumerate(column_transformer.transformers_): 
        transformer_name, transformer, orig_feature_names = transformer_item
        orig_feature_names = list(orig_feature_names)
        
        if transformer == 'drop':
            continue
            
        # if isinstance(transformer, Pipeline):
        #     # if pipeline, get the last transformer in the Pipeline
        #     transformer = transformer.steps[-1][1]


        if hasattr(transformer, 'get_feature_names'):
            if 'input_features' in transformer.get_feature_names.__code__.co_varnames:
                names = list(transformer.get_feature_names(orig_feature_names))
            else:
                names = list(transformer.get_feature_names())

        # elif hasattr(transformer, 'get_feature_names_out'):
        #     if 'input_features' in transformer.get_feature_names_out.__code__.co_varnames:
        #         names = list(transformer.get_feature_names_out(orig_feature_names))
        #     else:
        #         names = list(transformer.get_feature_names_out())

        elif hasattr(transformer,'indicator_') and transformer.add_indicator:
            # is this transformer one of the imputers & did it call the MissingIndicator?
            missing_indicator_indices = transformer.indicator_.features_
            missing_indicators = [orig_feature_names[idx] + '_missing_flag'\
                                    for idx in missing_indicator_indices]
            names = orig_feature_names + missing_indicators

        elif hasattr(transformer,'features_'):
            # is this a MissingIndicator class? 
            missing_indicator_indices = transformer.features_
            missing_indicators = [orig_feature_names[idx] + '_missing_flag'\
                                    for idx in missing_indicator_indices]

        else:
            names = orig_feature_names

        new_feature_names.extend(names)

    return new_feature_names

