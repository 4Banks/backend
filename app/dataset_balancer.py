import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN

SEED = 42

def random_under_sampling(df):
	X = df.drop('Class', axis=1)
	y = df['Class']

	under_sampler = RandomUnderSampler(random_state=SEED)
	X_under, y_under = under_sampler.fit_resample(X, y)
	df_under = pd.DataFrame(X_under)
	df_under['Class'] = y_under

	return df_under

def random_over_sampling(df):
	X = df.drop('Class', axis=1)
	y = df['Class']

	over_sampler = RandomOverSampler(random_state=SEED)
	X_over, y_over = over_sampler.fit_resample(X, y)
	df_over = pd.DataFrame(X_over)
	df_over['Class'] = y_over

	return df_over

def smote(df):
	X = df.drop('Class', axis=1)
	y = df['Class']

	smote = SMOTE(random_state=SEED, sampling_strategy='minority')
	X_smote, y_smote = smote.fit_resample(X, y)
	df_smote = pd.DataFrame(X_smote)
	df_smote['Class'] = y_smote

	return df_smote

def bsmote(df):
	X = df.drop('Class', axis=1)
	y = df['Class']

	b_smote = BorderlineSMOTE(random_state=SEED, sampling_strategy='minority')
	X_bsmote, y_bsmote = b_smote.fit_resample(X, y)
	df_bsmote = pd.DataFrame(X_bsmote)
	df_bsmote['Class'] = y_bsmote

	return df_bsmote

def adasyn(df):
	X = df.drop('Class', axis=1)
	y = df['Class']

	adasyn = ADASYN(random_state=SEED, sampling_strategy='minority')
	X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
	df_adasyn = pd.DataFrame(X_adasyn)
	df_adasyn['Class'] = y_adasyn

	return df_adasyn