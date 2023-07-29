import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.missing_data_treater import handle_missing_data

def test_handle_missing_data_remove():
    df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
    df_handled = handle_missing_data(df, 'remove')
    assert df_handled.isnull().sum().sum() == 0
    assert df_handled.shape[0] == 1

def test_handle_missing_data_mean():
    df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
    df_handled = handle_missing_data(df, 'mean')
    assert df_handled.isnull().sum().sum() == 0
    assert df_handled['A'][2] == 1.5
    assert df_handled['B'][1] == 5.0

def test_handle_missing_data_median():
    df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
    df_handled = handle_missing_data(df, 'median')
    assert df_handled.isnull().sum().sum() == 0
    assert df_handled['A'][2] == 1.5
    assert df_handled['B'][1] == 5.0

def test_handle_missing_data_most_frequent():
    df = pd.DataFrame({'A': [1, 2, np.nan, 1], 'B': [4, np.nan, 6, 4]})
    df_handled = handle_missing_data(df, 'most_frequent')
    assert df_handled.isnull().sum().sum() == 0
    assert df_handled['A'][2] == 1
    assert df_handled['B'][1] == 4

def test_handle_missing_data_constant():
    df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
    df_handled = handle_missing_data(df, 'constant', 0)
    assert df_handled.isnull().sum().sum() == 0
    assert df_handled['A'][2] == 0
    assert df_handled['B'][1] == 0
