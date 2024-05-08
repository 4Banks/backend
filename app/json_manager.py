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

def save_json(data: dict, dataset_id: str, file_name: str, to_gcs: bool = False) -> None:
    '''
    Esta função salva um dicionário como um arquivo JSON localmente ou em um bucket do Google Cloud Storage.

    ### Parâmetros:
    - `data` (dict, obrigatório): O dicionário a ser salvo.
    - `dataset_id` (str, obrigatório): O ID do dataset. O arquivo JSON será salvo localmente ou no bucket do Google Cloud Storage 
                                       sob o caminho `{dataset_id}/{file_name}.json`.
    - `file_name` (str, obrigatório): O nome do arquivo JSON.
    - `to_gcs` (bool, opcional): Se o arquivo JSON deve ser salvo no bucket do Google Cloud Storage. O padrão é `False`.

    ### Não retorna nada.

    ### Gera uma exceção:
    - `google.cloud.exceptions.GoogleCloudError`: Se ocorrer um erro ao tentar salvar o arquivo no bucket.
    '''
    if to_gcs:
        save_json_to_gcs(data, dataset_id, file_name)
    else:
        save_json_to_local(data, dataset_id, file_name)

def save_json_to_local(data: dict, dataset_id: str, file_name: str) -> None:
    '''
    Esta função salva um dicionário como um arquivo JSON localmente.

    ### Parâmetros:
    - `data` (dict, obrigatório): O dicionário a ser salvo.
    - `dataset_id` (str, obrigatório): O ID do dataset. O arquivo JSON será salvo no diretório `app/datasets/{dataset_id}/{file_name}.json`.
    - `file_name` (str, obrigatório): O nome do arquivo JSON.

    ### Não retorna nada.

    ### Gera uma exceção:
    - `FileNotFoundError`: Se ocorrer um erro ao tentar salvar o arquivo localmente.
    '''
    os.makedirs(f'app/datasets/{dataset_id}', exist_ok=True)

    with open(f'app/datasets/{dataset_id}/{file_name}.json', 'w') as file:
        json.dump(data, file, cls=numpy_encoder)

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