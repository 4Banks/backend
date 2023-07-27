from google.cloud import storage
import google.auth
from google.oauth2 import service_account
from sklearn.tree import export_graphviz
import graphviz
from time import sleep

import os
os.environ["PATH"] += os.pathsep + '/usr/local/bin'

BUCKET_NAME = 'banks-dev-392615.appspot.com'
CREDENTIALS_PATH = 'banks-dev-392615-7412df8a19f0.json'

def get_credentials():
    '''
    Tenta carregar as credenciais a partir do arquivo JSON. Se não conseguir,
    cai de volta para as credenciais padrão do Google Cloud.

    ### Retorno:
    - `google.auth.credentials.Credentials`: Objeto de credenciais do Google Cloud.
    '''
    try:
        return service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    except Exception as e:
        credentials, _ = google.auth.default()
        return credentials

def save_image_to_gcs(dataset_id: str, file_name: str) -> None:
    '''
    Salva uma imagem no Google Cloud Storage.

    ### Parâmetros:
    - `dataset_id (str)`: ID do dataset.
    - `file_name (str)`: Nome do arquivo.
    '''
    credentials = get_credentials()
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob_name = f'{dataset_id}/{file_name}'
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_name)

def create_decision_tree_image(model, features: list, file_name: str):
    '''
    Cria uma imagem de árvore de decisão a partir de um modelo.

    ### Parâmetros:
    - `model`: Modelo treinado.
    - `features (list)`: Lista de features.
    - `file_name (str)`: Nome do arquivo.
    '''
    class_names = [str(name) for name in model.classes_]

    dot_data = export_graphviz(model, out_file=None, 
                               feature_names=features, 
                               class_names=class_names, 
                               filled=True,
                               rounded=True, 
                               special_characters=True)
    graph = graphviz.Source(dot_data) 
    graph.format = 'png'
    graph.render(filename=file_name, cleanup=True)
    sleep(1)

def delete_decision_tree_image(file_name):
    '''
    Deleta uma imagem de árvore de decisão local.

    ### Parâmetros:
    - `file_name (str)`: Nome do arquivo.
    '''
    if os.path.exists(file_name):
        os.remove(file_name)