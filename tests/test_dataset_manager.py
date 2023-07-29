import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.dataset_manager import get_credentials, load_csv_from_gcs, save_df_to_gcs

DATASET_ID = 'test'
FILE_NAME = 'test'

def test_get_credentials():
    credentials = get_credentials()
    assert credentials is not None, "Failed to get credentials"

def test_load_csv_from_gcs():
    df = load_csv_from_gcs(DATASET_ID, FILE_NAME)
    assert isinstance(df, pd.DataFrame), "Failed to load CSV from GCS"
    assert not df.empty, "DataFrame is empty"