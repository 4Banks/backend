import pandas as pd
import pytest
from unittest.mock import Mock, patch
from google.cloud import storage
from pandas.testing import assert_frame_equal
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.dataset_manager import load_csv_from_gcs, save_df_to_gcs

@patch('google.cloud.storage.Client')
def test_load_csv_from_gcs(mock_client):
    # Dados esperados
    expected_data = pd.DataFrame({
        'Time': [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 7.0, 7.0, 9.0, 10.0],
        'V1': [-1.359807, 1.191857, -1.358354, -0.966272, -1.158233, -0.425966, 1.229658, -0.644269, -0.894286, -0.338262, 1.449044],
        'V2': [-0.072781, 0.266151, -1.340163, -0.185226, 0.877737, 0.960523, 0.141004, 1.417964, 0.286157, 1.119593, -1.176339],
        'V3': [2.536347, 0.166480, 1.773209, 1.792993, 1.548718, 1.141109, 0.045371, 1.074380, -0.113192, 1.044367, 0.913860],
        'V4': [1.378155, 0.448154, 0.379780, -0.863291, 0.403034, -0.168252, 1.202613, -0.492199, -0.271526, -0.222187, -1.375667],
    })

    # Mock do objeto blob e de sua função download_as_text
    mock_blob = Mock()
    mock_blob.download_as_text.return_value = expected_data.to_csv(index=False)

    # Mock do bucket
    mock_bucket = Mock()
    mock_bucket.blob.return_value = mock_blob

    # Configura o mock do cliente do GCS para retornar o mock do bucket
    mock_client.return_value.bucket.return_value = mock_bucket

    # Chama a função com o dataset_id 'test_id'
    actual_data = load_csv_from_gcs('test_id')

    # Verifica se o DataFrame retornado é o esperado
    assert_frame_equal(actual_data, expected_data)

    # Verifica se a função download_as_text foi chamada uma vez
    mock_blob.download_as_text.assert_called_once()

    # Verifica se a função blob foi chamada uma vez com o argumento correto
    mock_bucket.blob.assert_called_once_with('test_id/test_id.csv')

    # Verifica se a função bucket foi chamada uma vez com o argumento correto
    mock_client.return_value.bucket.assert_called_once_with('banks-dev-392615.appspot.com')

def test_save_df_to_gcs():
    # Cria um DataFrame para testar
    data = {'Time': [0.0, 1.0, 2.0],
            'V1': [-1.359807, 1.191857, -1.358354],
            'V2': [-0.072781, 0.266151, -1.340163],
            'V3': [2.536347, 0.166480, 1.773209],
            'V4': [1.378155, 0.448154, 0.379780]}
    df = pd.DataFrame(data)

    dataset_id = 'test_id'
    file_name = 'test_file'

    # Faz o mock da função upload_from_string para não fazer um upload real
    with patch.object(storage.Blob, 'upload_from_string') as mock_upload:
        save_df_to_gcs(df, dataset_id, file_name)
    
    # Verifica se a função upload_from_string foi chamada com os argumentos corretos
    mock_upload.assert_called_once_with(df.to_csv(index=False), 'text/csv')