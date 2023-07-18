from google.cloud import storage
import pandas as pd
from io import StringIO
from google.oauth2 import service_account

BUCKET_NAME = 'banks-dev-392615.appspot.com'

def load_csv_from_gcs(dataset_id: str) -> pd.DataFrame:
    '''
    Esta função baixa e carrega um arquivo CSV de um bucket do Google Cloud Storage.

    Parâmetros:
        dataset_id (str, obrigatório): O ID do dataset. O arquivo CSV correspondente a este dataset_id
                                       deve estar localizado no bucket do Google Cloud Storage sob o 
                                       caminho `{dataset_id}/{dataset_id}.csv`.

    Retorna:
        pd.DataFrame: Um DataFrame pandas contendo os dados do arquivo CSV baixado.

    Raises:
        google.cloud.exceptions.NotFound: Se o arquivo CSV correspondente ao dataset_id não for encontrado no bucket.
    '''
    credentials = service_account.Credentials.from_service_account_file('banks-dev-392615-80afa5d33712.json')
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob_name = f'{dataset_id}/{dataset_id}.csv'
    blob = bucket.blob(blob_name)
    
    blob_content_as_string = blob.download_as_text()
    data = pd.read_csv(StringIO(blob_content_as_string))

    return data

def save_df_to_gcs(df: pd.DataFrame, dataset_id: str, file_name: str) -> None:
    '''
    Esta função salva um DataFrame pandas como um arquivo CSV em um bucket do Google Cloud Storage.

    Parâmetros:
        df (pd.DataFrame, obrigatório): O DataFrame a ser salvo.
        dataset_id (str, obrigatório): O ID do dataset. O arquivo CSV será salvo no bucket do Google Cloud Storage 
                                       sob o caminho `{dataset_id}/{file_name}.csv`.
        file_name (str, obrigatório): O nome do arquivo CSV.

    Não retorna nada.

    Raises:
        google.cloud.exceptions.GoogleCloudError: Se ocorrer um erro ao tentar salvar o arquivo no bucket.
    '''
    credentials = service_account.Credentials.from_service_account_file('banks-dev-392615-80afa5d33712.json')
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob_name = f'{dataset_id}/{file_name}.csv'
    blob = bucket.blob(blob_name)

    # Convertemos o DataFrame em um CSV e o carregamos para o blob
    blob.upload_from_string(df.to_csv(index=False), 'text/csv')