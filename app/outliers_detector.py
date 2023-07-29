import pandas as pd
import numpy as np
from scipy.stats import zscore

def detect_outliers(df: pd.DataFrame,
                    z_score_method: bool = False,
                    robust_z_score_method: bool = False,
                    iqr_method: bool = False,
                    winsorization_method: bool = False) -> dict:
    '''
    Detecta outliers no dataset.

    ### Métodos:
    - Z-score
    - Robust Z-score
    - IQR
    - Winsorization

    ### Parâmetros:
    - `df`: DataFrame com os dados.
    - `z_score_method`: Ativa o método Z-score.
    - `robust_z_score_method`: Ativa o método Robust Z-score.
    - `iqr_method`: Ativa o método IQR.
    - `winsorization_method`: Ativa o método Winsorization.

    ### Retorno:outliers/
    - `dict` com os outliers detectados.
    '''
    df_outliers = df.copy()
    df_outliers = df_outliers.drop(columns=['Class'])

    methods = {
        'z_score': z_score_method,
        'robust_z_score': robust_z_score_method,
        'iqr': iqr_method,
        'winsorization': winsorization_method
    }
    outliers_dict = {}

    for method, active in methods.items():
        if active:
            outliers_dict[method] = {}

            for column in df_outliers.columns:
                if method == 'z_score':
                    z_scores = np.abs(zscore(df_outliers[column]))
                    outliers = np.where(z_scores > 3)
                    outliers_dict['z_score'][column] = outliers[0].tolist()

                if method == 'robust_z_score':
                    median = df_outliers[column].median()
                    mad = np.median(np.abs(df_outliers[column] - median))
                    modified_z_scores = 0.6745 * (df_outliers[column] - median) / mad
                    outliers = np.where(np.abs(modified_z_scores) > 3.5)
                    outliers_dict['robust_z_score'][column] = outliers[0].tolist()

                if method == 'iqr':
                    Q1 = df_outliers[column].quantile(0.25)
                    Q3 = df_outliers[column].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df_outliers[(df_outliers[column] < (Q1 - 1.5 * IQR)) | (df_outliers[column] > (Q3 + 1.5 * IQR))].index
                    outliers_dict['iqr'][column] = outliers.tolist()

                if method == 'winsorization':
                    q = df_outliers[column].quantile([0.01, 0.99])
                    outliers = df_outliers[(df_outliers[column] < q.iloc[0]) | (df_outliers[column] > q.iloc[1])].index
                    outliers_dict['winsorization'][column] = outliers.tolist()

    return outliers_dict