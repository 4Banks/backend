import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.outliers_detector import detect_outliers

def test_detect_outliers_z_score():
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 100], 'B': [6, 7, 8, 9, 10, 200], 'Class': [0, 0, 0, 0, 0, 1]})
    outliers_dict = detect_outliers(df, z_score_method=True)
    assert outliers_dict['z_score']['A'] == []
    assert outliers_dict['z_score']['B'] == []

def test_detect_outliers_robust_z_score():
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 100], 'B': [6, 7, 8, 9, 10, 200], 'Class': [0, 0, 0, 0, 0, 1]})
    outliers_dict = detect_outliers(df, robust_z_score_method=True)
    assert outliers_dict['robust_z_score']['A'] == [5]
    assert outliers_dict['robust_z_score']['B'] == [5]

def test_detect_outliers_iqr():
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 100], 'B': [6, 7, 8, 9, 10, 200], 'Class': [0, 0, 0, 0, 0, 1]})
    outliers_dict = detect_outliers(df, iqr_method=True)
    assert outliers_dict['iqr']['A'] == [5]
    assert outliers_dict['iqr']['B'] == [5]

def test_detect_outliers_winsorization():
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 100], 'B': [6, 7, 8, 9, 10, 200], 'Class': [0, 0, 0, 0, 0, 1]})
    outliers_dict = detect_outliers(df, winsorization_method=True)
    assert outliers_dict['winsorization']['A'] == [0, 5]
    assert outliers_dict['winsorization']['B'] == [0, 5]
