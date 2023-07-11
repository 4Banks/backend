from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

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