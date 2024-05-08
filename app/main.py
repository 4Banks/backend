from app.machine_learning import train_and_evaluate_model, training_tasks
from app.outliers_treater import transform_outliers
from app.outliers_detector import detect_outliers
from app.superficial_analysis import generate_statistics, generate_correlation_matrix
from app.missing_data_treater import handle_missing_data
from app.dataset_balancer import random_under_sampling, random_over_sampling, smote, bsmote, adasyn
from app.json_manager import save_json
from app.dataset_manager import load_csv, save_df
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from google.api_core.exceptions import NotFound
from threading import Thread
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '.')))

USE_GCS = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', 'http://34.172.61.208:3000'],
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
            df = load_csv(dataset_id=dataset_id,
                          file_name=file_name, index=0, from_gcs=USE_GCS)
        else:
            df = load_csv(dataset_id=dataset_id,
                          file_name=file_name, from_gcs=USE_GCS)
        return JSONResponse(content=df.head().to_dict(orient='records'))
    except NotFound:
        raise HTTPException(
            status_code=404, detail=f'Dataset "{dataset_id}/{file_name}" não encontrado no bucket')


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
        df = load_csv(dataset_id=dataset_id, file_name=file_name,
                      index=0, from_gcs=USE_GCS)
    else:
        df = load_csv(dataset_id=dataset_id,
                      file_name=file_name, from_gcs=USE_GCS)

    df = generate_statistics(df)

    save_df(df, dataset_id,
            f'{file_name}_superficial_analysis', index=True, to_gcs=USE_GCS)

    if USE_GCS:
        gcs_path = f'gs://<BUCKET_NAME>/{dataset_id}/{file_name}_superficial_analysis.csv'
        return JSONResponse(content={'message': f'Resultado salvo com sucesso no seguinte local: {gcs_path}'})

    local_path = f'app/datasets/{dataset_id}/{file_name}_superficial_analysis.csv'
    return JSONResponse(content={'message': f'Resultado salvo com sucesso no seguinte local: {local_path}'})


@app.get('/correlations/{dataset_id}/{file_name}/', response_description='Calcula a correlação entre os atributos de um dataset',)
def get_correlations(dataset_id: str,
                     file_name: str,
                     index: bool = False,
                     correlation_pearson: bool = False,
                     correlation_kendall: bool = False,
                     correlation_spearman: bool = False) -> float:
    '''
    Esta função carrega os dados de um dataset a partir do bucket do Google Cloud Storage,
    calcula a correlação entre os atributos do dataset e salva os resultados no Google Cloud Storage.

    ### Parâmetros:
    - `dataset_id` (str, obrigatório): O ID do dataset. O arquivo CSV correspondente a este dataset_id
                                        deve estar localizado no bucket do Google Cloud Storage sob o
                                        caminho `{dataset_id}/{file_name}.csv`.
    - `file_name` (str, obrigatório): O nome do arquivo CSV.
    - `index` (bool, opcional): Se o DataFrame possui índice a ser carregado. O padrão é `False`.
    - `correlation_pearson` (bool, opcional): Se a correlação de Pearson deve ser calculada. O padrão é `False`.
    - `correlation_kendall` (bool, opcional): Se a correlação de Kendall deve ser calculada. O padrão é `False`.
    - `correlation_spearman` (bool, opcional): Se a correlação de Spearman deve ser calculada. O padrão é `False`.

    ### Retorna:
    - `JSONResponse`: Um JSONResponse onde o conteúdo é uma lista de dicionários com as correlações calculadas.
    '''
    correlations = {
        'pearson': correlation_pearson,
        'kendall': correlation_kendall,
        'spearman': correlation_spearman
    }

    if correlation_pearson or correlation_kendall or correlation_spearman:
        if index:
            df = load_csv(dataset_id=dataset_id,
                          file_name=file_name, index=0, from_gcs=USE_GCS)
        else:
            df = load_csv(dataset_id=dataset_id,
                          file_name=file_name, from_gcs=USE_GCS)

        correlations_matrixes = generate_correlation_matrix(
            df, correlation_pearson, correlation_kendall, correlation_spearman)

        for correlation_index, correlation_name in enumerate(correlations):
            if correlations[correlation_name]:
                save_df(correlations_matrixes[correlation_index], dataset_id,
                        f'{file_name}_correlation_{correlation_name}', index=True, to_gcs=USE_GCS)

        if USE_GCS:
            gcs_path = f'gs://<BUCKET_NAME>/{dataset_id}/{file_name}_correlation_<correlation_name>.csv'
            return JSONResponse(content={'message': f'Resultados salvos com sucesso no seguinte local: {gcs_path}'})

        local_path = f'app/datasets/{dataset_id}/{file_name}_correlation_<correlation_name>.csv'
        return JSONResponse(content={'message': f'Resultado salvo com sucesso no seguinte local: {local_path}'})

    return JSONResponse(content={'message': 'Nenhuma correlação foi calculada'})


@app.get('/outliers_detect_and_transform/{dataset_id}/{file_name}/', response_description='Detecta outliers em um dataset',)
def detect_and_transform_dataset_outliers(dataset_id: str,
                                          file_name: str,
                                          index: bool = False,
                                          z_score: bool = False,
                                          robust_z_score: bool = False,
                                          iqr: bool = False,
                                          winsorization: bool = False,
                                          treatment_method: str = None,
                                          treatment_constant_value: float = None) -> JSONResponse:
    '''
    Esta função carrega os dados de um dataset a partir do bucket do Google Cloud Storage,
    detecta outliers, trata os outliers e retorna o resultado.

    Se o arquivo CSV correspondente ao `dataset_id` não for encontrado no bucket,
    a função retorna um código de status HTTP 404 e uma mensagem de erro personalizada.

    ### Parâmetros:
    - `dataset_id` (str, obrigatório): O ID do dataset. O arquivo CSV correspondente a este dataset_id
                                        deve estar localizado no bucket do Google Cloud Storage sob o
                                        caminho `{dataset_id}/{file_name}.csv`.
    - `file_name` (str, obrigatório): O nome do arquivo CSV.
    - `index` (bool, opcional): Se o DataFrame possui índice a ser carregado. O padrão é `False`.
    - `z_score` (bool, opcional): Se o método Z-score deve ser utilizado para detecção de outliers. O padrão é `False`.
    - `robust_z_score` (bool, opcional): Se o método Robust Z-score deve ser utilizado para detecção de outliers. O padrão é `False`.
    - `iqr` (bool, opcional): Se o método IQR deve ser utilizado para detecção de outliers. O padrão é `False`.
    - `winsorization` (bool, opcional): Se o método Winsorization deve ser utilizado para detecção de outliers. O padrão é `False`.
    - `treatment_method` (str, opcional): O método de tratamento de outliers a ser utilizado. Os valores possíveis são:
        - log
        - sqrt
        - cbrt
        - scaling
        - constant
        - remove
    - `treatment_constant_value` (float, opcional): O valor constante a ser utilizado no método de tratamento `constant`.
                                                    O padrão é `0`.

    ### Retorna:
    - `JSONResponse`: Um JSONResponse onde o conteúdo é um dicionário com a mensagem de que o arquivo
                      foi salvo com sucesso e o caminho do arquivo no Google Cloud Storage.

    ### Gera uma exceção:
    - `HTTPException`: Se o arquivo CSV correspondente ao dataset_id não for encontrado no bucket.
                          A exceção contém um código de status HTTP 404 e uma mensagem detalhada.
    - `HTTPException`: Se o método de tratamento de outliers não for encontrado.
                          A exceção contém um código de status HTTP 400 e uma mensagem detalhada.                 
    '''
    if treatment_method not in ['log', 'sqrt', 'cbrt', 'scaling', 'constant', 'remove', None]:
        raise HTTPException(
            status_code=400, detail=f'Método "{treatment_method}" não encontrado')

    if index:
        df = load_csv(dataset_id=dataset_id, file_name=file_name, index=0)
    else:
        df = load_csv(dataset_id=dataset_id, file_name=file_name)

    outliers_dict = detect_outliers(
        df, z_score, robust_z_score, iqr, winsorization)
    df = transform_outliers(
        df, outliers_dict, treatment_method, treatment_constant_value)

    if USE_GCS:
        save_json(outliers_dict, dataset_id, f'{file_name}_outliers')
        outliers_path = f'gs://<BUCKET_NAME>/{dataset_id}/{file_name}_outliers.json'
        df_path = f'gs://<BUCKET_NAME>/{dataset_id}/{file_name}_outliers_treated.csv'
    else:
        save_df(df, dataset_id, f'{file_name}_outliers_treated', index=index)
        outliers_path = f'app/datasets/{dataset_id}/{file_name}_outliers.json'
        df_path = f'app/datasets/{dataset_id}/{file_name}_outliers_treated.csv'

    return JSONResponse(content={'message': f'Outliers detectados e tratados com sucesso. Resultados salvos nos seguintes locais: {outliers_path} e {df_path}'})


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
        df = load_csv(dataset_id=dataset_id, file_name=file_name,
                      index=0, from_gcs=USE_GCS)
    else:
        df = load_csv(dataset_id=dataset_id, file_name=file_name)

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
        raise HTTPException(
            status_code=400, detail=f'Método "{method}" não encontrado')

    save_df(df, dataset_id, f'{file_name}_{method}', index=index)

    if USE_GCS:
        path = f'gs://<BUCKET_NAME>/{dataset_id}/{file_name}_{method}.csv'
    else:
        path = f'app/datasets/{dataset_id}/{file_name}_{method}.csv'
    return {'message': f'Resultado salvo com sucesso no seguinte local: {path}'}


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
        raise HTTPException(
            status_code=400, detail=f'Classificador "{classifier}" não encontrado')

    if USE_GCS:
        path = f'gs://<BUCKET_NAME>/{dataset_id}/{file_name}_{classifier}.json'
        if classifier == 'decision_tree':
            path += f' e gs://<BUCKET_NAME>/{dataset_id}/{file_name}_{classifier}.png'
    else:
        path = f'app/datasets/{dataset_id}/{file_name}_{classifier}.json'
        if classifier == 'decision_tree':
            path += f' e app/datasets/{dataset_id}/{file_name}_{classifier}.png'

    return JSONResponse(content={'message': f'O treinamento do classificador "{classifier}" foi iniciado. O resultado será salvo no seguinte local: {path}'})


@app.get('/running_training_tasks/{dataset_id}', response_description='Retorna uma lista com os treinamentos em andamento',)
def get_dataset_running_training_tasks(dataset_id: str) -> JSONResponse:
    '''
    Esta função retorna uma lista com os treinamentos em andamento.

    ### Parâmetros:
    - `dataset_id` (str, obrigatório): O ID do dataset.

    ### Retorna:
    - `JSONResponse`: Um JSONResponse onde o conteúdo é uma lista com os treinamentos em andamento.
    '''
    running_training_tasks = []
    for task in training_tasks:
        if task.startswith(dataset_id) and training_tasks[task]['status'] == 'running':
            running_training_tasks.append(training_tasks[task])
    if len(running_training_tasks) == 0:
        return JSONResponse(content={'message': 'Não há treinamentos em andamento'})
    print(f'Running tasks: {running_training_tasks}')
    return JSONResponse(content=running_training_tasks)


@app.get('/finished_training_tasks/{dataset_id}', response_description='Retorna uma lista com os treinamentos finalizados',)
def get_dataset_finished_training_tasks(dataset_id: str) -> JSONResponse:
    '''
    Esta função retorna uma lista com os treinamentos finalizados.

    ### Parâmetros:
    - `dataset_id` (str, obrigatório): O ID do dataset.

    ### Retorna:
    - `JSONResponse`: Um JSONResponse onde o conteúdo é uma lista com os treinamentos finalizados.
    '''
    finished_training_tasks = []
    for task in training_tasks:
        if task.startswith(dataset_id) and training_tasks[task]['status'] == 'finished':
            finished_training_tasks.append(training_tasks[task])
    if len(finished_training_tasks) == 0:
        return JSONResponse(content={'message': 'Não há treinamentos finalizados'})
    print(f'Finished tasks: {finished_training_tasks}')
    return JSONResponse(content=finished_training_tasks)


@app.get('/failed_training_tasks/{dataset_id}', response_description='Retorna uma lista com os treinamentos que falharam',)
def get_dataset_failed_training_tasks(dataset_id: str) -> JSONResponse:
    '''
    Esta função retorna uma lista com os treinamentos que falharam.

    ### Parâmetros:
    - `dataset_id` (str, obrigatório): O ID do dataset.

    ### Retorna:
    - `JSONResponse`: Um JSONResponse onde o conteúdo é uma lista com os treinamentos que falharam.
    '''
    failed_training_tasks = []
    for task in training_tasks:
        if task.startswith(dataset_id) and training_tasks[task]['status'] == 'failed':
            failed_training_tasks.append(training_tasks[task])
    if len(failed_training_tasks) == 0:
        return JSONResponse(content={'message': 'Não há treinamentos que falharam'})
    return JSONResponse(content=failed_training_tasks)


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
                     outliers_treatment_constant_value: float = 0,
                     balance_method: str = None,
                     superficial_analysis: bool = False,
                     correlation_pearson: bool = False,
                     correlation_kendall: bool = False,
                     correlation_spearman: bool = False,
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
                                                             O padrão é `0`.
    - `balance_method` (str, opcional): O método de balanceamento a ser utilizado. Os valores possíveis são:
        - random_under_sampling
        - random_over_sampling
        - smote
        - bsmote
        - adasyn
    - `superficial_analysis` (bool, opcional): Se a análise superficial deve ser executada. O padrão é `False`.
    - `correlation_pearson` (bool, opcional): Se a correlação de Pearson deve ser calculada. O padrão é `False`.
    - `correlation_kendall` (bool, opcional): Se a correlação de Kendall deve ser calculada. O padrão é `False`.
    - `correlation_spearman` (bool, opcional): Se a correlação de Spearman deve ser calculada. O padrão é `False`.
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
        df = load_csv(dataset_id=dataset_id, file_name=file_name,
                      index=0, from_gcs=USE_GCS)
    else:
        df = load_csv(dataset_id=dataset_id,
                      file_name=file_name, from_gcs=USE_GCS)
    print('Dados carregados com sucesso')

    if missing_data_method is not None:
        print('Iniciando tratamento de dados faltantes...', end=' ')
        df = handle_missing_data(
            df, missing_data_method, missing_data_constant_value)
        print('Tratamento de dados faltantes finalizado')

    if outliers_treatment_method is not None:
        print('Iniciando tratamento de outliers...', end=' ')
        outliers_dict = detect_outliers(
            df, outliers_z_score, outliers_robust_z_score, outliers_iqr, outliers_winsorization)
        df = transform_outliers(
            df, outliers_dict, outliers_treatment_method, outliers_treatment_constant_value)
        print('Tratamento de outliers finalizado')

    if balance_method is not None:
        print(f'Iniciando balanceamento "{balance_method}"...', end=' ')
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
            raise HTTPException(
                status_code=400, detail=f'Método "{balance_method}" não encontrado')
        print('Balanceamento finalizado')

    if superficial_analysis:
        print('Iniciando análise superficial...', end=' ')
        df_superficial_analysis = generate_statistics(df.copy())
        save_df(df_superficial_analysis, dataset_id,
                f'{file_name}_superficial_analysis', index=True, to_gcs=USE_GCS)
        print('Análise superficial finalizada')

    if correlation_pearson or correlation_kendall or correlation_spearman:
        correlations = {
            'pearson': correlation_pearson,
            'kendall': correlation_kendall,
            'spearman': correlation_spearman
        }
        print('Iniciando cálculo de correlações...', end=' ')
        correlations_matrixes = generate_correlation_matrix(
            df, correlation_pearson, correlation_kendall, correlation_spearman)
        for correlation_index, correlation_name in enumerate(correlations):
            if correlations[correlation_name]:
                save_df(correlations_matrixes[correlation_index], dataset_id,
                        f'{file_name}_correlation_{correlation_name}', index=True, to_gcs=USE_GCS)
        print('Cálculo de correlações finalizado')

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

    message = 'Pipeline finalizado com sucesso.'
    if USE_GCS:
        message += f' Os resultados serão salvos no seguinte local: gs://<BUCKET_NAME>/{dataset_id}/'
        return JSONResponse(content={'message': message, 'use_gcs': True})

    message += f' Os resultados serão salvos no seguinte local: app/datasets/{dataset_id}/'
    return JSONResponse(content={'message': message, 'use_gcs': False})


@app.post("/upload/{dataset_id}/")
async def upload_file(dataset_id: str, file: UploadFile = File(...)):
    os.makedirs(f"app/datasets/{dataset_id}", exist_ok=True)
    file_location = f"app/datasets/{dataset_id}/{file.filename}"

    try:
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"info": "File saved successfully", "file_path": file_location}


@app.get('/datasets/{dataset_id}/{file_name}')
def serve_file(dataset_id: str, file_name: str):
    file_path = f'app/datasets/{dataset_id}/{file_name}'
    if os.path.exists(file_path):
        return FileResponse(path=file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")


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
