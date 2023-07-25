from google.cloud import storage
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from dataset_manager import get_credentials, BUCKET_NAME

class numpy_encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_json_to_gcs(data: dict, dataset_id: str, file_name: str) -> None:
    '''
    Esta função salva um dicionário como um arquivo JSON em um bucket do Google Cloud Storage e torna o arquivo carregado público.

    ### Parâmetros:
    - `data` (dict, obrigatório): O dicionário a ser salvo.
    - `dataset_id` (str, obrigatório): O ID do dataset. O arquivo JSON será salvo no bucket do Google Cloud Storage 
                                       sob o caminho `{dataset_id}/{file_name}.json`.
    - `file_name` (str, obrigatório): O nome do arquivo JSON.

    ### Não retorna nada.

    ### Gera uma exceção:
    - `google.cloud.exceptions.GoogleCloudError`: Se ocorrer um erro ao tentar salvar o arquivo no bucket.
    '''
    credentials = get_credentials()
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob_name = f'{dataset_id}/{file_name}.json'
    blob = bucket.blob(blob_name)

    blob.upload_from_string(json.dumps(data, cls=numpy_encoder), 'application/json')