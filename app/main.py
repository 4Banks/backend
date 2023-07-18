from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from dataset_manager import load_csv_from_gcs
from fastapi import HTTPException
from google.api_core.exceptions import NotFound

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
    """
    Rota inicial da API
    Este endpoint retorna uma mensagem de boas vindas
    """
    return JSONResponse(content={'message': '4Banks API!'})

@app.get('/datasets/{dataset_id}', response_description="Carrega os dados de um dataset",)
def load_dataset(dataset_id: str):
    '''
    Esta função carrega os dados de um dataset a partir do bucket do Google Cloud Storage. 

    Se o arquivo CSV correspondente ao dataset_id não for encontrado no bucket, 
    a função retorna um código de status HTTP 404 e uma mensagem de erro personalizada.

    Parâmetros:
        dataset_id (str, obrigatório): O ID do dataset. O arquivo CSV correspondente a este dataset_id
                                       deve estar localizado no bucket do Google Cloud Storage sob o 
                                       caminho `{dataset_id}/{dataset_id}.csv`.

    Retorna:
        JSONResponse: Um JSONResponse onde o conteúdo é uma lista de registros do arquivo CSV baixado.
                      Cada registro é um dicionário onde a chave é o nome da coluna e o valor é o valor da célula.

    Raises:
        HTTPException: Se o arquivo CSV correspondente ao dataset_id não for encontrado no bucket.
                       A exceção contém um código de status HTTP 404 e uma mensagem detalhada.
    '''
    try:
        df = load_csv_from_gcs(dataset_id)
        return JSONResponse(content=df.to_dict(orient='records'))
    except NotFound:
        raise HTTPException(status_code=404, detail=f'Dataset "{dataset_id}" não encontrado no bucket')

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Your application name",
        version="1.0.0",
        description="This is a very custom OpenAPI schema",
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