from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi import HTTPException
from google.api_core.exceptions import NotFound
from threading import Thread
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from dataset_manager import load_csv_from_gcs, save_df_to_gcs
from json_manager import save_json_to_gcs
from dataset_balancer import random_under_sampling, random_over_sampling, smote, bsmote, adasyn
from missing_data_treater import handle_missing_data
from superficial_analysis import generate_statistics
from outliers_detector import detect_outliers
from outliers_treater import transform_outliers
from machine_learning import train_and_evaluate_model, training_tasks

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/', response_description='Retorna a mensagem de boas vindas',)
def hello():
    '''
    Rota inicial da API.

    ### Retorna:
    - `JSONResponse`: Um JSONResponse com uma mensagem de boas vindas.
    '''
    return JSONResponse(content={'message': '4banks API!'})

@app.get('/dataset/{dataset_id}/{file_name}', response_description='Carrega os dados de um dataset',)
def load_dataset(dataset_id: str,
                 file_name: str,
                 index: bool = False):
    '''
    Esta função carrega os dados de um dataset a partir do bucket do Google Cloud Storage. 

    Se o arquivo CSV correspondente ao dataset_id não for encontrado no bucket, 
    a função retorna um código de status HTTP 404 e uma mensagem de erro personalizada.

    ### Parâmetros:
    - `dataset_id` (str, obrigatório): O ID do dataset. O arquivo CSV correspondente a este dataset_id
                                        deve estar localizado no bucket do Google Cloud Storage sob o
                                        caminho `{dataset_id}/{file_name}.csv`.
    - `file_name` (str, obrigatório): O nome do arquivo CSV.
    - `index` (bool, opcional): Se o DataFrame possui índice a ser carregado. O padrão é `False`.

    ### Retorna:
    - `JSONResponse`: Um JSONResponse onde o conteúdo é uma lista de registros do arquivo CSV baixado.
                      Cada registro é um dicionário onde a chave é o nome da coluna e o valor é o valor da célula.

    ### Gera uma exceção:
    - `HTTPException`: Se o arquivo CSV correspondente ao dataset_id não for encontrado no bucket.
                       A exceção contém um código de status HTTP 404 e uma mensagem detalhada.
    '''
    try:
        if index:
            df = load_csv_from_gcs(dataset_id=dataset_id, file_name=file_name, index=0)
        else:
            df = load_csv_from_gcs(dataset_id=dataset_id, file_name=file_name)
        return JSONResponse(content=df.head().to_dict(orient='records'))
    except NotFound:
        raise HTTPException(status_code=404, detail=f'Dataset "{dataset_id}/{file_name}" não encontrado no bucket')

@app.get('/superficial_analysis/{dataset_id}/{file_name}/', response_description='Gera estatísticas superficiais sobre os dados de um dataset',)
def generate_superficial_analysis(dataset_id: str,
                                  file_name: str,
                                  index: bool = False) -> JSONResponse:
    '''
    Esta função carrega os dados de um dataset a partir do bucket do Google Cloud Storage,
    gera estatísticas superficiais sobre os dados e retorna o resultado.

    Se o arquivo CSV correspondente ao `dataset_id` não for encontrado no bucket,
    a função retorna um código de status HTTP 404 e uma mensagem de erro personalizada.

    ### Parâmetros:
    - `dataset_id` (str, obrigatório): O ID do dataset. O arquivo CSV correspondente a este dataset_id
                                       deve estar localizado no bucket do Google Cloud Storage sob o
                                       caminho `{dataset_id}/{file_name}.csv`.
    - `file_name` (str, obrigatório): O nome do arquivo CSV.
    - `index` (bool, opcional): Se o DataFrame possui índice a ser carregado. O padrão é `False`.

    ### Retorna:
    - `JSONResponse`: Um JSONResponse onde o conteúdo é um dicionário com a mensagem de que o arquivo
                      foi salvo com sucesso e o caminho do arquivo no Google Cloud Storage.

    ### Gera uma exceção:
    - `HTTPException`: Se o arquivo CSV correspondente ao dataset_id não for encontrado no bucket.
                       A exceção contém um código de status HTTP 404 e uma mensagem detalhada.
    '''
    if index:
        df = load_csv_from_gcs(dataset_id=dataset_id, file_name=file_name, index=0)
    else:
        df = load_csv_from_gcs(dataset_id=dataset_id, file_name=file_name)

    df = generate_statistics(df)

    save_df_to_gcs(df, dataset_id, f'{file_name}_superficial_analysis', index=True)

    gcs_path = f'gs://<BUCKET_NAME>/{dataset_id}/{file_name}_superficial_analysis.csv'

    return JSONResponse(content={'message': f'Resultado salvo com sucesso no seguinte local: {gcs_path}'})

@app.get('/outliers/{dataset_id}/{file_name}/detect', response_description='Detecta outliers em um dataset',)
def get_dataset_outliers(dataset_id: str,
                         file_name: str,
                         index: bool = False) -> JSONResponse:
    if index:
        df = load_csv_from_gcs(dataset_id=dataset_id, file_name=file_name, index=0)
    else:
        df = load_csv_from_gcs(dataset_id=dataset_id, file_name=file_name)

    outliers_dict = detect_outliers(df)

    save_json_to_gcs(outliers_dict, dataset_id, f'{file_name}_outliers')

    gcs_path = f'gs://<BUCKET_NAME>/{dataset_id}/{file_name}_outliers.json'

    return JSONResponse(content={'message': f'Resultado salvo com sucesso no seguinte local: {gcs_path}'})

@app.get('/balance/{dataset_id}/{file_name}', response_description="Balanceia os dados de um dataset",)
def balance_dataset(dataset_id: str,
                    file_name: str,
                    method: str,
                    index: bool = False):
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
    - `index` (bool, opcional): Se o DataFrame possui índice a ser carregado. O padrão é `False`.

    ### Retorna:
    - `JSONResponse`: Um JSONResponse onde o conteúdo é um dicionário com a mensagem de que o arquivo 
                      foi salvo com sucesso e o caminho do arquivo no Google Cloud Storage.
    
    ### Gera uma exceção:
    - `HTTPException`: Se o arquivo CSV correspondente ao dataset_id não for encontrado no bucket.
                       A exceção contém um código de status HTTP 404 e uma mensagem detalhada.
    - `HTTPException`: Se o método de balanceamento não for encontrado.
                       A exceção contém um código de status HTTP 400 e uma mensagem detalhada.
    '''
    if index:
        df = load_csv_from_gcs(dataset_id=dataset_id, file_name=file_name, index=0)
    else:
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

    gcs_path = f'gs://<BUCKET_NAME>/{dataset_id}/{file_name}_{method}.csv'

    return {'message': f'Resultado salvo com sucesso no seguinte local: {gcs_path}'}

@app.get('/machine_learning/{dataset_id}/{file_name}/{classifier}', response_description='Aplica um algoritmo de Machine Learning em um dataset',)
def apply_machine_learning(classifier: str,
                           dataset_id: str,
                           file_name: str,
                           index: bool = False):
    '''
    Esta função carrega os dados de um dataset a partir do bucket do Google Cloud Storage,
    aplica um algoritmo de aprendizado de máquina e retorna as métricas de teste, a matriz
    de confusão do teste e a importância ordenada dos atributos.

    Se o arquivo CSV correspondente ao `dataset_id` não for encontrado no bucket,
    a função retorna um código de status HTTP 404 e uma mensagem de erro personalizada.

    ### Parâmetros:
    - `classifier` (str, obrigatório): O classificador a ser utilizado. Os valores possíveis são:
        - logistic_regression
        - decision_tree
        - random_forest
        - xgboost
        - lightgbm
        - mlp
    - `dataset_id` (str, obrigatório): O ID do dataset. O arquivo CSV correspondente a este dataset_id
                                        deve estar localizado no bucket do Google Cloud Storage sob o
                                        caminho `{dataset_id}/{file_name}.csv`.
    - `file_name` (str, obrigatório): O nome do arquivo CSV.
    - `index` (bool, opcional): Se o DataFrame possui índice a ser carregado. O padrão é `False`.

    ### Retorna:
    - `JSONResponse`: Um JSONResponse onde o conteúdo é um dicionário com as métricas de teste, a matriz
                        de confusão do teste e a importância ordenada dos atributos.

    ### Gera uma exceção:
    - `HTTPException`: Se o arquivo CSV correspondente ao dataset_id não for encontrado no bucket.
                       A exceção contém um código de status HTTP 404 e uma mensagem detalhada.
    - `HTTPException`: Se o classificador não for encontrado.
                       A exceção contém um código de status HTTP 400 e uma mensagem detalhada.
    '''    
    if classifier in ['logistic_regression', 'decision_tree', 'random_forest', 'xgboost', 'lightgbm', 'mlp']:
        Thread(target=train_and_evaluate_model, kwargs={
            'dataset_id': dataset_id,
            'file_name': file_name,
            'model_name': classifier,
            'index': index}).start()
    else:
        raise HTTPException(status_code=400, detail=f'Classificador "{classifier}" não encontrado')

    gcs_path = f'gs://<BUCKET_NAME>/{dataset_id}/{file_name}_{classifier}.json'

    return JSONResponse(content={'message': f'O treinamento do classificador "{classifier}" foi iniciado. O resultado será salvo no seguinte local: {gcs_path}'})

@app.get('/training_status/{dataset_id}/{model_name}', response_description='Verifica o status de um treinamento',)
def check_training_status(dataset_id: str, model_name: str) -> JSONResponse:
    '''
    Esta função verifica o status de um treinamento.

    ### Parâmetros:
    - `dataset_id` (str, obrigatório): O ID do dataset.
    - `model_name` (str, obrigatório): O nome do modelo. Possíveis valores:	
        - logistic_regression
        - decision_tree
        - random_forest
        - xgboost
        - lightgbm
        - mlp

    ### Retorna:
    - `JSONResponse`: Um JSONResponse onde o conteúdo é um dicionário com o status do treinamento.
    '''
    if f'{dataset_id}_{model_name}' in training_tasks:
        return JSONResponse(content={'status': training_tasks[f'{dataset_id}_{model_name}']})
    else:
        return JSONResponse(content={'status': 'Treinamento não iniciado'})

@app.get('/pipeline/{dataset_id}/{file_name}', response_description='Executa o pipeline completo de análise de dados',)
def execute_pipeline(dataset_id: str,
                     file_name: str,
                     index: bool = False,
                     missing_data_method: str = None,
                     missing_data_constant_value: float = None,
                     outliers_z_score: bool = False,
                     outliers_robust_z_score: bool = False,
                     outliers_iqr: bool = False,
                     outliers_winsorization: bool = False,
                     outliers_treatment_method: str = None,
                     outliers_treatment_constant_value: float = None,
                     balance_method: str = None,
                     superficial_analysis: bool = False,
                     ml_logistic_regression: bool = False,
                     ml_decision_tree: bool = False,
                     ml_random_forest: bool = False,
                     ml_xgboost: bool = False,
                     ml_lightgbm: bool = False,
                     ml_mlp: bool = False) -> JSONResponse:
    '''
    Esta função executa o pipeline completo de análise de dados.

    ### Parâmetros:
    - `dataset_id` (str, obrigatório): O ID do dataset. O arquivo CSV correspondente a este dataset_id
                                       deve estar localizado no bucket do Google Cloud Storage sob o
                                       caminho `{dataset_id}/{file_name}.csv`.
    - `file_name` (str, obrigatório): O nome do arquivo CSV.
    - `index` (bool, opcional): Se o DataFrame possui índice a ser carregado. O padrão é `False`.
    - `missing_data_method` (str, opcional): O método de tratamento de dados faltantes a ser utilizado. Os valores possíveis são:
        - remove
        - mean
        - median
        - most_frequent
        - constant
    - `missing_data_constant_value` (float, opcional): O valor constante a ser utilizado no método de tratamento `constant`.
                                                       O padrão é `None`.
    - `outliers_z_score` (bool, opcional): Se o método Z-score deve ser utilizado para detecção de outliers. O padrão é `False`.
    - `outliers_robust_z_score` (bool, opcional): Se o método Robust Z-score deve ser utilizado para detecção de outliers. O padrão é `False`.
    - `outliers_iqr` (bool, opcional): Se o método IQR deve ser utilizado para detecção de outliers. O padrão é `False`.
    - `outliers_winsorization` (bool, opcional): Se o método Winsorization deve ser utilizado para detecção de outliers. O padrão é `False`.
    - `outliers_treatment_method` (str, opcional): O método de tratamento de outliers a ser utilizado. Os valores possíveis são:
        - log
        - sqrt
        - cbrt
        - scaling
        - constant
        - remove
    - `outliers_treatment_constant_value` (float, opcional): O valor constante a ser utilizado no método de tratamento `constant`.
                                                             O padrão é `None`.
    - `balance_method` (str, opcional): O método de balanceamento a ser utilizado. Os valores possíveis são:
        - random_under_sampling
        - random_over_sampling
        - smote
        - bsmote
        - adasyn
    - `superficial_analysis` (bool, opcional): Se a análise superficial deve ser executada. O padrão é `False`.
    - `ml_logistic_regression` (bool, opcional): Se a regressão logística deve ser executada. O padrão é `False`.
    - `ml_decision_tree` (bool, opcional): Se a árvore de decisão deve ser executada. O padrão é `False`.
    - `ml_random_forest` (bool, opcional): Se a floresta aleatória deve ser executada. O padrão é `False`.
    - `ml_xgboost` (bool, opcional): Se o XGBoost deve ser executado. O padrão é `False`.
    - `ml_lightgbm` (bool, opcional): Se o LightGBM deve ser executado. O padrão é `False`.
    - `ml_mlp` (bool, opcional): Se a rede neural MLP deve ser executada. O padrão é `False`.

    ### Retorna:
    - `JSONResponse`: Um JSONResponse onde o conteúdo é um dicionário com a mensagem de que o pipeline foi executado com sucesso.
    '''
    print('Iniciando pipeline...')
    if index:
        df = load_csv_from_gcs(dataset_id=dataset_id, file_name=file_name, index=0)
    else:
        df = load_csv_from_gcs(dataset_id=dataset_id, file_name=file_name)
    print('Dados carregados com sucesso')

    print('Iniciando tratamento de dados faltantes...')
    if missing_data_method is not None:
        df = handle_missing_data(df, missing_data_method, missing_data_constant_value)
    print('Tratamento de dados faltantes finalizado')

    print('Iniciando tratamento de outliers...')
    if outliers_treatment_method is not None:
        outliers_dict = detect_outliers(df)
        df = transform_outliers(df,
                                outliers_dict,
                                outliers_z_score,
                                outliers_robust_z_score,
                                outliers_iqr,
                                outliers_winsorization,
                                outliers_treatment_method,
                                outliers_treatment_constant_value)
    print('Tratamento de outliers finalizado')

    print('Iniciando balanceamento...')
    if balance_method is not None:
        if balance_method == 'random_under_sampling':
            df = random_under_sampling(df)
        elif balance_method == 'random_over_sampling':
            df = random_over_sampling(df)
        elif balance_method == 'smote':
            df = smote(df)
        elif balance_method == 'bsmote':
            df = bsmote(df)
        elif balance_method == 'adasyn':
            df = adasyn(df)
        else:
            raise HTTPException(status_code=400, detail=f'Método "{balance_method}" não encontrado')
    print('Balanceamento finalizado')

    print('Iniciando análise superficial...')
    if superficial_analysis:
        df_superficial_analysis = generate_statistics(df)
        save_df_to_gcs(df_superficial_analysis, dataset_id, f'{file_name}_superficial_analysis', index=True)
    print('Análise superficial finalizada')

    print('Iniciando treinamento dos modelos...')
    if ml_logistic_regression:
        Thread(target=train_and_evaluate_model, kwargs={
            'dataset_id': dataset_id,
            'file_name': file_name,
            'model_name': 'logistic_regression',
            'df': df,
            'index': index}).start()
        
    if ml_decision_tree:
        Thread(target=train_and_evaluate_model, kwargs={
            'dataset_id': dataset_id,
            'file_name': file_name,
            'model_name': 'decision_tree',
            'df': df,
            'index': index}).start()
        
    if ml_random_forest:
        Thread(target=train_and_evaluate_model, kwargs={
            'dataset_id': dataset_id,
            'file_name': file_name,
            'model_name': 'random_forest',
            'df': df,
            'index': index}).start()

    if ml_xgboost:
        Thread(target=train_and_evaluate_model, kwargs={
            'dataset_id': dataset_id,
            'file_name': file_name,
            'model_name': 'xgboost',
            'df': df,
            'index': index}).start()

    if ml_lightgbm:
        Thread(target=train_and_evaluate_model, kwargs={
            'dataset_id': dataset_id,
            'file_name': file_name,
            'model_name': 'lightgbm',
            'df': df,
            'index': index}).start()

    if ml_mlp:
        Thread(target=train_and_evaluate_model, kwargs={
            'dataset_id': dataset_id,
            'file_name': file_name,
            'model_name': 'mlp',
            'df': df,
            'index': index}).start()
    print('Treinamentos inicializados')

    return JSONResponse(content={'message': 'Pipeline finalizada com sucesso.'})

def custom_openapi():
    '''
    Função que cria e retorna o esquema OpenAPI personalizado.

    ### Retorna:
    - `dict`: O esquema OpenAPI personalizado.
    '''
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title='4banks API',
        version='1.0.0',
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)