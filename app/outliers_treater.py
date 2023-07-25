from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

def transform_outliers(df: pd.DataFrame,
                       outliers_dict: dict,
                       z_score: bool = False,
                       robust_z_score: bool = False,
                       iqr: bool = False,
                       winsorization: bool = False,
                       treatment_method=None,
                       treatment_constant_value=None):
    
    df_transformed = df.copy()
    scaler = MinMaxScaler()

    methods = {
        'z_score': z_score,
        'robust_z_score': robust_z_score,
        'iqr': iqr,
        'winsorization': winsorization
    }

    attributes = df_transformed.columns.drop('Class')

    for method, active in methods.items():
        if active:
            for attribute in attributes:
                if attribute in outliers_dict[method]:
                    outlier_indices = outliers_dict[method][attribute]
                    valid_indices = df_transformed.index.intersection(outlier_indices)

                    if treatment_method == 'log':
                        df_transformed[attribute] = df_transformed[attribute].map(lambda x: np.log(x) if x > 0 else 0)

                    elif treatment_method == 'sqrt':
                        df_transformed[attribute] = df_transformed[attribute].map(lambda x: np.sqrt(x) if x > 0 else 0)

                    elif treatment_method == 'cbrt':
                        df_transformed[attribute] = df_transformed[attribute].map(lambda x: np.cbrt(x) if x > 0 else 0)

                    elif treatment_method == 'scaling':
                        df_transformed[attribute] = scaler.fit_transform(df_transformed[attribute].values.reshape(-1,1))

                    elif treatment_method == 'constant':
                        df_transformed.loc[valid_indices, attribute] = treatment_constant_value

                    elif treatment_method == 'remove':
                        df_transformed = df_transformed.drop(valid_indices)

                    else:
                        print('Nenhuma transformação de outliers selecionada')

    return df_transformed