from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import sys
import os
import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from dataset_manager import load_csv
from json_manager import save_json
from image_manager import create_decision_tree_image, save_image_to_gcs, delete_decision_tree_image

SEED = 42
training_tasks = {}

def calculate_max_iter(df_length: int, base_iter: int = 200, scale_factor: float = 0.05) -> int:
    if df_length < 100:
        return base_iter
    return int(base_iter + scale_factor * np.log(df_length) * base_iter)

def start_training_task(dataset_id: str, model_name: str) -> None:
    print(f'Started training task for dataset {dataset_id} and model {model_name}')
    training_tasks[f'{dataset_id}_{model_name}'] = {
        'dataset_id': dataset_id,
        'model_name': model_name,
        'status': 'running',
        'start_time': datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
    }

def finish_training_task(dataset_id: str, model_name: str) -> None:
    print(f'Finished training task for dataset {dataset_id} and model {model_name}')
    training_tasks[f'{dataset_id}_{model_name}']['status'] = 'finished'
    training_tasks[f'{dataset_id}_{model_name}']['finish_time'] = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

def failed_training_task(dataset_id: str, model_name: str) -> None:
    print(f'Failed training task for dataset {dataset_id} and model {model_name}')
    training_tasks[f'{dataset_id}_{model_name}']['status'] = 'failed'
    training_tasks[f'{dataset_id}_{model_name}']['finish_time'] = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

def load_dataset(dataset_id: str, file_name: str, index: bool = False) -> pd.DataFrame:
    if index:
        return load_csv(dataset_id, file_name, index_col=0)
    else:
        return load_csv(dataset_id, file_name)

def get_selected_model(model:str, max_iter) -> object:
    if model == 'logistic_regression':
        return LogisticRegression(random_state=SEED)
    elif model == 'decision_tree':
        return DecisionTreeClassifier(random_state=SEED)
    elif model == 'random_forest':
        return RandomForestClassifier(random_state=SEED)
    elif model == 'xgboost':
        return XGBClassifier(random_state=SEED)
    elif model == 'lightgbm':
        return LGBMClassifier(random_state=SEED)
    elif model == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(100, 50, 25),
                             alpha=0.01,
                             solver='adam',
                             random_state=SEED,
                             learning_rate='adaptive',
                             learning_rate_init=0.01,
                             max_iter=max_iter,
                             n_iter_no_change=int(0.15*max_iter),
                             verbose=True)

def train_and_evaluate_model(dataset_id: str,
                             file_name: str,
                             model_name: str,
                             df: pd.DataFrame = None,
                             index: bool = False) -> dict:
    
    start_training_task(dataset_id, model_name)

    try:
        if df is None:
            df = load_dataset(dataset_id, file_name, index=index)
        
        max_iter = calculate_max_iter(df_length=len(df))
        
        y = df['Class']

        X = df.drop(columns=['Class'])

        #scaler = StandardScaler()
        #X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=SEED)
        
        model = get_selected_model(model_name, max_iter=max_iter)
        model.fit(X_train, y_train)

        if model_name == 'decision_tree':
            create_decision_tree_image(model, X.columns.tolist(), f'app/datasets/{dataset_id}/{file_name}_decision_tree')
            # save_image(dataset_id, f'{file_name}_decision_tree.png')
            # delete_decision_tree_image(f'{file_name}_decision_tree.png')

        y_pred = model.predict(X_test)
        
        metrics = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics['accuracy'] = accuracy_score(y_test, y_pred, normalize=True)

        cm = confusion_matrix(y_test, y_pred)

        importance = permutation_importance(model, X_test, y_test, n_repeats=10)
        feature_importance = np.mean(importance.importances, axis=1)
        feature_importance_ranking = {name: importance for name, importance in sorted(zip(X.columns, feature_importance), key=lambda x: x[1], reverse=True)}
        
        result = {
            'performance_metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_importance_ranking,
        }

        save_json(result, dataset_id, f'{file_name}_{model_name}')
        finish_training_task(dataset_id, model_name)
    except Exception as e:
        failed_training_task(dataset_id, model_name)
        raise e