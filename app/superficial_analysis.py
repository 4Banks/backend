import pandas as pd
import numpy as np
from scipy.stats import mode

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