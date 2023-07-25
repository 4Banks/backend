import pandas as pd
import numpy as np
from scipy.stats import zscore

def detect_outliers(df: pd.DataFrame) -> dict:
    '''
    Detecta outliers no dataset.

    ### Métodos:
    - Z-score
    - Robust Z-score
    - IQR
    - Winsorization

    ### Parâmetros:
    - `df`: DataFrame com os dados.

    ### Retorno:
    - `dict` com os outliers detectados.
    '''
    df_outliers = df.copy()
    df_outliers = df_outliers.drop(columns=['Class'])

    methods = ["z_score", "robust_z_score", "iqr", "winsorization"]
    outliers_dict = {method: {} for method in methods}

    for column in df_outliers:
        # Método Z-score
        z_scores = np.abs(zscore(df_outliers[column]))
        outliers = np.where(z_scores > 3)
        outliers_dict["z_score"][column] = outliers[0].tolist()

        # Método Robust Z-score
        median = df_outliers[column].median()
        mad = np.median(np.abs(df_outliers[column] - median))
        modified_z_scores = 0.6745 * (df_outliers[column] - median) / mad
        outliers = np.where(np.abs(modified_z_scores) > 3.5)
        outliers_dict["robust_z_score"][column] = outliers[0].tolist()

        # Método IQR
        Q1 = df_outliers[column].quantile(0.25)
        Q3 = df_outliers[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df_outliers[(df_outliers[column] < (Q1 - 1.5 * IQR)) | (df_outliers[column] > (Q3 + 1.5 * IQR))].index
        outliers_dict["iqr"][column] = outliers.tolist()

        # Método Winsorization
        q = df_outliers[column].quantile([0.01, 0.99])
        outliers = df_outliers[(df_outliers[column] < q.iloc[0]) | (df_outliers[column] > q.iloc[1])].index
        outliers_dict["winsorization"][column] = outliers.tolist()

    return outliers_dict