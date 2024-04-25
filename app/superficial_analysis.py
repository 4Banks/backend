import pandas as pd
import numpy as np
from scipy.stats import mode
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def generate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Gera estatísticas superficiais sobre o dataset.

    ### Parâmetros:
    - `df`: DataFrame com os dados.
    
    ### Retorno:
        - `DataFrame` com as estatísticas.
    '''
    results = {}

    columns = df.columns.copy()
    columns = columns.drop('Class')

    for column in columns:
        column_data = df[column]
        column_mean = column_data.mean()
        column_median = column_data.median()
        column_mode = mode(column_data, keepdims=True)[0][0]
        num_missing = column_data.isnull().sum()
        percent_missing = (num_missing / len(df)) * 100
        num_zeros = (column_data == 0).sum()
        max_value = column_data.max()
        min_value = column_data.min()
        std_dev = column_data.std()
        range_value = np.ptp(column_data)
        iqr = np.percentile(column_data, 75) - np.percentile(column_data, 25)
        skewness = column_data.skew()
        kurtosis = column_data.kurtosis()

        results[column] = {
            'Média': column_mean,
            'Mediana': column_median,
            'Moda': column_mode,
            'Campos vazios': num_missing,
            'Campos vazios (%)': percent_missing,
            'Campos com valor zero': num_zeros,
            'Valor máximo': max_value,
            'Valor mínimo': min_value,
            'Desvio padrão': std_dev,
            'Intervalo de valores': range_value,
            'IQR': iqr,
            'Assimetria': skewness,
            'Curtose': kurtosis
        }

    results_df = pd.DataFrame.from_dict(results, orient='index')
    
    return results_df.transpose()

def generate_correlation_matrix(df: pd.DataFrame,
                                correlation_pearson: bool = False,
                                correlation_kendall: bool = False,
                                correlation_spearman: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if correlation_pearson:
        print('Calculando correlação de Pearson...')
        correlation_pearson_matrix = df.corr(method='pearson')
    if correlation_kendall:
        print('Calculando correlação de Kendall...')
        correlation_kendall_matrix = df.corr(method='kendall')
    if correlation_spearman:
        print('Calculando correlação de Spearman...')
        correlation_spearman_matrix = df.corr(method='spearman')

    return (correlation_pearson_matrix, correlation_kendall_matrix, correlation_spearman_matrix)