from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi import HTTPException
from google.api_core.exceptions import NotFound
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from dataset_manager import load_csv_from_gcs, save_df_to_gcs
from dataset_balancer import random_under_sampling, random_over_sampling, smote, bsmote, adasyn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/', response_description="Retorna a mensagem de boas vindas",)
def hello():
    '''
    Rota inicial da API.

    ### Retorna:
    - `JSONResponse`: Um JSONResponse com uma mensagem de boas vindas.
    '''
    return JSONResponse(content={'message': '4Banks API!'})

@app.get('/datasets/{dataset_id}/{file_name}', response_description="Carrega os dados de um dataset",)
def load_dataset(dataset_id: str, file_name: str):
    '''
    Esta função carrega os dados de um dataset a partir do bucket do Google Cloud Storage. 

    Se o arquivo CSV correspondente ao dataset_id não for encontrado no bucket, 
    a função retorna um código de status HTTP 404 e uma mensagem de erro personalizada.

    ### Parâmetros:
    - `dataset_id` (str, obrigatório): O ID do dataset. O arquivo CSV correspondente a este dataset_id
                                        deve estar localizado no bucket do Google Cloud Storage sob o
                                        caminho `{dataset_id}/{file_name}.csv`.
    - `file_name` (str, obrigatório): O nome do arquivo CSV.

    ### Retorna:
    - `JSONResponse`: Um JSONResponse onde o conteúdo é uma lista de registros do arquivo CSV baixado.
                      Cada registro é um dicionário onde a chave é o nome da coluna e o valor é o valor da célula.

    ### Gera uma exceção:
    - `HTTPException`: Se o arquivo CSV correspondente ao dataset_id não for encontrado no bucket.
                       A exceção contém um código de status HTTP 404 e uma mensagem detalhada.
    '''
    try:
        df = load_csv_from_gcs(dataset_id=dataset_id, file_name=file_name)
        return JSONResponse(content=df.to_dict(orient='records'))
    except NotFound:
        raise HTTPException(status_code=404, detail=f'Dataset "{dataset_id}/{file_name}" não encontrado no bucket')

@app.get('/datasets/{dataset_id}/{file_name}/balance', response_description="Balanceia os dados de um dataset",)
def balance_dataset(dataset_id: str, file_name: str, method: str):
    '''
    Esta função carrega os dados de um dataset a partir do bucket do Google Cloud Storage,
    balanceia os dados e retorna o resultado.

    Se o arquivo CSV correspondente ao `dataset_id` não for encontrado no bucket,
    a função retorna um código de status HTTP 404 e uma mensagem de erro personalizada.

    ### Parâmetros:
    - `dataset_id` (str, obrigatório): O ID do dataset. O arquivo CSV correspondente a este dataset_id
                                        deve estar localizado no bucket do Google Cloud Storage sob o
                                        caminho `{dataset_id}/{file_name}.csv`.
    - `file_name` (str, obrigatório): O nome do arquivo CSV.
    - `method` (str, obrigatório): O método de balanceamento a ser utilizado. Os valores possíveis são:
        - random_under_sampling
        - random_over_sampling
        - smote
        - bsmote
        - adasyn

    ### Retorna:
    - `JSONResponse`: Um JSONResponse onde o conteúdo é um dicionário com a mensagem de que o arquivo 
                      foi salvo com sucesso e o caminho do arquivo no Google Cloud Storage.
    
    ### Gera uma exceção:
    - `HTTPException`: Se o arquivo CSV correspondente ao dataset_id não for encontrado no bucket.
                       A exceção contém um código de status HTTP 404 e uma mensagem detalhada.
    - `HTTPException`: Se o método de balanceamento não for encontrado.
                       A exceção contém um código de status HTTP 400 e uma mensagem detalhada.
    '''
    df = load_csv_from_gcs(dataset_id=dataset_id, file_name=file_name)

    if method == 'random_under_sampling':
        df = random_under_sampling(df)
    elif method == 'random_over_sampling':
        df = random_over_sampling(df)
    elif method == 'smote':
        df = smote(df)
    elif method == 'bsmote':
        df = bsmote(df)
    elif method == 'adasyn':
        df = adasyn(df)
    else:
        raise HTTPException(status_code=400, detail=f'Método "{method}" não encontrado')
    
    save_df_to_gcs(df, dataset_id, f'{file_name}_{method}')

    gcs_path = f"gs://<BUCKET_NAME>/{dataset_id}/{file_name}_{method}.csv"

    return {"message": f"Resultado salvo com sucesso no seguinte local: {gcs_path}"}

def custom_openapi():
    '''
    Função que cria e retorna o esquema OpenAPI personalizado.

    ### Retorna:
    - `dict`: O esquema OpenAPI personalizado.
    '''
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="4banks API",
        version="1.0.0",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "/flasgger_static/swagger-ui/logo_small.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)