from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

def handle_missing_data(df, method, constant_value=None):
    '''
    Esta função trata os dados faltantes de um DataFrame.

    ### Parâmetros:
    - `df` (DataFrame, obrigatório): O DataFrame a ser tratado.
    - `method` (str, obrigatório): O método de tratamento a ser utilizado. Os valores possíveis são:
        - remove
        - mean
        - median
        - most_frequent
        - constant
    - `constant_value` (int ou float, opcional): O valor constante a ser utilizado no método de tratamento `constant`.
                                                 O padrão é `None`.

    ### Retorna:
    - `DataFrame`: O DataFrame tratado.

    ### Gera uma exceção:
    - `ValueError`: Se o método de tratamento não for encontrado.
    - `ValueError`: Se o valor constante não for fornecido para o método de tratamento `constant`.
    '''
    df_handled = df.copy()

    if method == 'remove':
        df_handled.dropna(inplace=True)

    elif method == 'mean':
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        df_handled = pd.DataFrame(imputer.fit_transform(df_handled), columns=df.columns)

    elif method == 'median':
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        df_handled = pd.DataFrame(imputer.fit_transform(df_handled), columns=df.columns)

    elif method == 'most_frequent':
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        df_handled = pd.DataFrame(imputer.fit_transform(df_handled), columns=df.columns)

    elif method == 'constant' and constant_value is not None:
        imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=constant_value)
        df_handled = pd.DataFrame(imputer.fit_transform(df_handled), columns=df.columns)

    else:
        print('Método inválido ou valor constante não foi fornecido para inputação constante')

    return df_handled