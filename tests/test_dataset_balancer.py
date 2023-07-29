import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.dataset_balancer import apply_resampler, random_under_sampling, random_over_sampling, smote, bsmote, adasyn
from imblearn.under_sampling import RandomUnderSampler

SEED = 42
np.random.seed(SEED)
df = pd.DataFrame({
    'Class': np.random.choice([0, 1], size=(1000,), p=[2./3, 1./3]),
    'Feature 1': np.random.normal(0, 1, 1000),
    'Feature 2': np.random.normal(0, 2, 1000),
    'Feature 3': np.random.normal(0, 3, 1000),
    'Feature 4': np.random.normal(0, 4, 1000),
    'Feature 5': np.random.normal(0, 5, 1000),
})
resampler = RandomUnderSampler(random_state=SEED)

def test_apply_resampler():
    df_resampled = apply_resampler(df, resampler=resampler)
    assert df_resampled.shape[0] <= df.shape[0]
    assert df_resampled['Class'].value_counts()[0] == df_resampled['Class'].value_counts()[1]

def test_random_over_sampling():
    df_resampled = random_over_sampling(df)
    assert df_resampled.shape[0] >= df.shape[0]
    assert df_resampled['Class'].value_counts()[0] == df_resampled['Class'].value_counts()[1]

def test_random_under_sampling():
    df_resampled = random_under_sampling(df)
    assert df_resampled.shape[0] <= df.shape[0]
    assert df_resampled['Class'].value_counts()[0] == df_resampled['Class'].value_counts()[1]

def test_smote():
    df_resampled = smote(df)
    assert df_resampled.shape[0] >= df.shape[0]
    assert df_resampled['Class'].value_counts()[0] == df_resampled['Class'].value_counts()[1]

def test_bsmote():
    df_resampled = bsmote(df)
    assert df_resampled.shape[0] >= df.shape[0]
    assert df_resampled['Class'].value_counts()[0] == df_resampled['Class'].value_counts()[1]

def test_adasyn():
    df_resampled = adasyn(df)
    assert df_resampled.shape[0] >= df.shape[0]
    assert abs(df_resampled['Class'].value_counts()[0] - df_resampled['Class'].value_counts()[1]) <= 5
