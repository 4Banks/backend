import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN

SEED = 42

def apply_resampler(df: pd.DataFrame, resampler) -> pd.DataFrame:
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_resampled, y_resampled = resampler.fit_resample(X, y)
    df_resampled = pd.DataFrame(X_resampled)
    df_resampled['Class'] = y_resampled

    return df_resampled

def random_under_sampling(df: pd.DataFrame) -> pd.DataFrame:
    under_sampler = RandomUnderSampler(random_state=SEED)
    return apply_resampler(df, under_sampler)

def random_over_sampling(df: pd.DataFrame) -> pd.DataFrame:
    over_sampler = RandomOverSampler(random_state=SEED)
    return apply_resampler(df, over_sampler)

def smote(df: pd.DataFrame) -> pd.DataFrame:
    smote_sampler = SMOTE(random_state=SEED, sampling_strategy='minority')
    return apply_resampler(df, smote_sampler)

def bsmote(df: pd.DataFrame) -> pd.DataFrame:
    bsmote_sampler = BorderlineSMOTE(random_state=SEED, sampling_strategy='minority')
    return apply_resampler(df, bsmote_sampler)

def adasyn(df: pd.DataFrame) -> pd.DataFrame:
    adasyn_sampler = ADASYN(random_state=SEED, sampling_strategy='minority')
    return apply_resampler(df, adasyn_sampler)