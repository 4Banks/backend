import pandas as pd
from unittest.mock import Mock, patch
from google.cloud import storage
from pandas.testing import assert_frame_equal
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.dataset_manager import load_csv_from_gcs, save_df_to_gcs

@patch('google.cloud.storage.Client')
def test_load_csv_from_gcs(mock_client):
    expected_data = pd.DataFrame({
        'V1': [-1.359807, 1.191857, -1.358354, -0.966272, -1.158233, -0.425966],
        'V2': [-0.072781, 0.266151, -1.340163, -0.185226, 0.877737, 0.960523],
        'V3': [2.536347, 0.166480, 1.773209, 1.792993, 1.548718, 1.141109],
        'V4': [1.378155, 0.448154, 0.379780, -0.863291, 0.403034, -0.168252],
        'Class': [0, 0, 0, 0, 1, 1]
    })

    mock_blob = Mock()
    mock_blob.download_as_text.return_value = expected_data.to_csv(index=False)

    mock_bucket = Mock()
    mock_bucket.blob.return_value = mock_blob

    mock_client.return_value.bucket.return_value = mock_bucket

    actual_data = load_csv_from_gcs(dataset_id='test_id', file_name='test_file_0')

    assert_frame_equal(actual_data, expected_data)

    mock_blob.download_as_text.assert_called_once()

    mock_bucket.blob.assert_called_once_with('test_id/test_file_0.csv')

    mock_client.return_value.bucket.assert_called_once_with('banks-dev-392615.appspot.com')

def test_save_df_to_gcs():
    df = pd.DataFrame({
        'V1': [-1.359807, 1.191857, -1.358354, -0.966272, -1.158233, -0.425966],
        'V2': [-0.072781, 0.266151, -1.340163, -0.185226, 0.877737, 0.960523],
        'V3': [2.536347, 0.166480, 1.773209, 1.792993, 1.548718, 1.141109],
        'V4': [1.378155, 0.448154, 0.379780, -0.863291, 0.403034, -0.168252],
        'Class': [0, 0, 0, 0, 1, 1]
    })

    dataset_id = 'test_id'
    file_name = 'test_file'

    with patch.object(storage.Blob, 'upload_from_string') as mock_upload:
        save_df_to_gcs(df, dataset_id, file_name)

    mock_upload.assert_called_once_with(df.to_csv(index=False), 'text/csv')