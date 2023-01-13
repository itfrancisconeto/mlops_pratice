from pytest import fixture
import pandas as pd
import numpy as np
import math
from os import path
from sklearn.tree import DecisionTreeClassifier 

import app

@fixture
def create_ml_object()->object:
    object = app.MachineLearningModel()
    return object

def test_must_return_a_dataframe_object(create_ml_object)->any:    
    df = create_ml_object.data_exploration()
    assert isinstance(df, pd.DataFrame)

def test_dataframe_must_contain_rows(create_ml_object)->any:
    df = create_ml_object.data_exploration()
    assert len(df.index) > 0

def test_check_null_values(create_ml_object)->any:
    df = pd.DataFrame(np.nan, index=[0, 1, 2, 3], columns=['A'])  
    assert create_ml_object.check_null_values(df) == True

def test_must_return_a_decision_tree_model_object(create_ml_object)->any:    
    model_object = create_ml_object.decision_tree_classifier()
    assert isinstance(model_object, DecisionTreeClassifier)

def test_train_test_definition_must_contain_30_percent_rows(create_ml_object)->any:
    model_object = create_ml_object.decision_tree_classifier()
    df = create_ml_object.data_exploration()
    y_test, y_pred, model_fit = create_ml_object.train_test_definition(df,model_object)
    assert len(y_test) == math.ceil(len(df.index)*.3)
    assert len(y_pred) == math.ceil(len(df.index)*.3)
    assert (model_fit is None) == False

def test_fit_model_file_exists(create_ml_object)->any:
    assert path.exists("../api/predictor.pkl") == True