# train.py

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, 
                            precision_score, roc_auc_score, f1_score)
from catboost import CatBoostClassifier
import json
import joblib
from datetime import datetime
from skimpy import skim
from skimpy import clean_columns

# Настройка путей
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / 'data' / 'diabetes.csv'
MODEL_DIR = BASE_DIR / 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save_model():
    # Загрузка и подготовка данных
    print("Загрузка данных...")
    df = pd.read_csv(DATA_PATH)
    df = clean_columns(df)
    
    # Разделение данных
    X = df.drop('outcome', axis=1)
    y = df['outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        stratify=y, 
        test_size=0.2, 
        random_state=42
    )
    
    # Обучение модели
    print("Обучение CatBoost модели...")
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        eval_metric='Recall',
        random_seed=42,
        verbose=100
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50
    )
    
    # Оценка модели
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Сохранение модели и метрик
    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"catboost_diabetes_{model_version}"
    
    print(f" Сохранение модели {model_name}...")
    joblib.dump(model, MODEL_DIR / f"{model_name}.joblib")
    
    with open(MODEL_DIR / f"{model_name}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\nОбучение завершено!")
    print("Метрики модели:")
    for k, v in metrics.items():
        if k == 'timestamp':
            print(f"{k.upper()}: {v}")
        else:
            print(f"{k.upper()}: {v:.4f}")
    
    return model, metrics

if __name__ == "__main__":
    train_and_save_model()