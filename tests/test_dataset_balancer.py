import numpy as np
import pandas as pd
from collections import Counter
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.dataset_balancer import random_under_sampling, random_over_sampling, smote, bsmote, adasyn

def create_unbalanced_df():
    np.random.seed(42)
    class0 = np.random.rand(900, 10)
    class1 = np.random.rand(100, 10)
    df0 = pd.DataFrame(class0, columns=[f'feat{i}' for i in range(10)])
    df0['Class'] = 0
    df1 = pd.DataFrame(class1, columns=[f'feat{i}' for i in range(10)])
    df1['Class'] = 1
    df = pd.concat([df0, df1])
    return df

df = create_unbalanced_df()

def test_random_under_sampling():
    df_under = random_under_sampling(df)
    assert Counter(df_under['Class'])[0] == Counter(df_under['Class'])[1]

def test_random_over_sampling():
    df_over = random_over_sampling(df)
    assert Counter(df_over['Class'])[0] == Counter(df_over['Class'])[1]

def test_smote():
    df_smote = smote(df)
    assert Counter(df_smote['Class'])[0] == Counter(df_smote['Class'])[1]

def test_bsmote():
    df_bsmote = bsmote(df)
    assert Counter(df_bsmote['Class'])[0] == Counter(df_bsmote['Class'])[1]

def test_adasyn():
    df_adasyn = adasyn(df)
    assert Counter(df_adasyn['Class'])[0] == Counter(df_adasyn['Class'])[1]
