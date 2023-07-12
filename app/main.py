from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import pandas as pd
from io import StringIO

app = FastAPI()

@app.get('/', response_description="Retorna a mensagem de boas vindas", 
    responses={
        200: {
            'description': 'Mensagem de boas vindas retornada com sucesso',
            'content': {
                'application/json': {
                    'example': {
                        'message': '4Banks API!'
                    }
                }
            }
        }
    }
)
def hello():
    """
    Rota inicial da API
    Este endpoint retorna uma mensagem de boas vindas
    """
    return JSONResponse(content={'message': '4Banks API!'})

@app.post("/uploadcsv/")
async def create_upload_file(file: UploadFile = File(...)):
    '''
    Esta função lida com solicitações POST para carregar um arquivo CSV.
    
    Parâmetros: 
        file (UploadFile, obrigatório): Um arquivo CSV para upload. O campo deve ser incluído no corpo da solicitação POST.
                                        O arquivo CSV é lido e decodificado em um DataFrame pandas.
    
    Retorna: 
        dict: Um dicionário contendo:
            filename (str): O nome do arquivo carregado.
            rows (int): O número de linhas no arquivo CSV carregado.
            columns (int): O número de colunas no arquivo CSV carregado.
    '''
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    return {
        "filename": file.filename,
        "rows": df.shape[0],
        "columns": df.shape[1]
    }

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