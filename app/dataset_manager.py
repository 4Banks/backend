from google.cloud import storage
import google.auth
import pandas as pd
from io import StringIO
from google.oauth2 import service_account

BUCKET_NAME = 'banks-dev-392615.appspot.com'
CREDENTIALS_PATH = 'banks-dev-392615-80afa5d33712.json'

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
        print(f"Failed to load credentials from file, falling back to default credentials: {e}")
        credentials, _ = google.auth.default()
        return credentials

def load_csv_from_gcs(dataset_id: str, file_name: str) -> pd.DataFrame:
    '''
    Esta função baixa e carrega um arquivo CSV de um bucket do Google Cloud Storage.

    ### Parâmetros:
    - `dataset_id` (str, obrigatório): O ID do dataset. O arquivo CSV correspondente a este dataset_id
                                        deve estar localizado no bucket do Google Cloud Storage sob o
                                        caminho `{dataset_id}/{file_name}.csv`.
    - `file_name` (str, obrigatório): O nome do arquivo CSV.

    ### Retorna:
    - `pd.DataFrame`: Um DataFrame pandas contendo os dados do arquivo CSV baixado.

    ### Gera uma exceção:
    - `google.cloud.exceptions.NotFound`: Se o arquivo CSV correspondente ao dataset_id não for encontrado no bucket.
    '''
    credentials = get_credentials()
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob_name = f'{dataset_id}/{file_name}.csv'
    blob = bucket.blob(blob_name)
    
    blob_content_as_string = blob.download_as_text()
    data = pd.read_csv(StringIO(blob_content_as_string))

    print(data.head())

    return data

def save_df_to_gcs(df: pd.DataFrame, dataset_id: str, file_name: str) -> None:
    '''
    Esta função salva um DataFrame pandas como um arquivo CSV em um bucket do Google Cloud Storage e torna o arquivo carregado público.

    ### Parâmetros:
    - `df` (pd.DataFrame, obrigatório): O DataFrame a ser salvo.
    - `dataset_id` (str, obrigatório): O ID do dataset. O arquivo CSV será salvo no bucket do Google Cloud Storage 
                                       sob o caminho `{dataset_id}/{file_name}.csv`.
    - `file_name` (str, obrigatório): O nome do arquivo CSV.

    ### Não retorna nada.

    ### Gera uma exceção:
    - `google.cloud.exceptions.GoogleCloudError`: Se ocorrer um erro ao tentar salvar o arquivo no bucket.
    '''
    credentials = get_credentials()
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob_name = f'{dataset_id}/{file_name}.csv'
    blob = bucket.blob(blob_name)

    blob.upload_from_string(df.to_csv(index=False), 'text/csv')