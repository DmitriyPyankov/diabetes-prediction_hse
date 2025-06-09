# predict.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import subprocess
from datetime import datetime
from typing import Dict, Union
from skimpy import skim
from skimpy import clean_columns


BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
TRAIN_SCRIPT = BASE_DIR / 'src' / 'train.py'

def load_latest_model() -> tuple:
    """Загружает последнюю обученную модель и её метрики"""
    model_files = sorted(MODEL_DIR.glob('catboost_diabetes_*.joblib'))
    
    if not model_files:
        print("Модель не найдена. Запуск обучения...")
        result = subprocess.run(['python', str(TRAIN_SCRIPT)], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError("Ошибка при запуске train.py")
        model_files = sorted(MODEL_DIR.glob('catboost_diabetes_*.joblib'))
        if not model_files:
            raise FileNotFoundError("Обучение завершилось, но модель не была сохранена.")
    
    latest_model = model_files[-1] 
    model_name = latest_model.stem
    metrics_file = MODEL_DIR / f"{model_name}_metrics.json"
    
    model = joblib.load(latest_model)
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    print(f"Загружена модель: {model_name}")
    
    return model, metrics

def predict_diabetes(model, input_data: Union[pd.DataFrame, Dict]) -> dict:
    """Выполняет предсказание на новых данных"""
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()
    
    proba = model.predict_proba(input_df)[:, 1][0]
    prediction = int(proba >= 0.5)
    
    return {
        'prediction': prediction,
        'probability': float(proba),
        'class': 'diabetes' if prediction == 1 else 'no diabetes',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

if __name__ == "__main__":
    try:
        # 1. Загрузка модели (или обучение при необходимости)
        model, metrics = load_latest_model()
        
        # 2. Ввод пользователя
        test_data = {
            'pregnancies': int(input("Введите количество беременностей: ")),
            'glucose': int(input("Введите уровень глюкозы: ")),
            'blood_pressure': int(input("Введите артериальное давление: ")),
            'skin_thickness': int(input("Введите толщину кожной складки (в мм): ")),
            'insulin': int(input("Введите уровень инсулина: ")),
            'bmi': float(input("Введите индекс массы тела (BMI): ")),
            'diabetes_pedigree_function': float(input("Введите показатель наследственности диабета: ")),
            'age': int(input("Введите возраст: "))
        }
        
        # 3. Предсказание
        result = predict_diabetes(model, test_data)
        
        print("Результат предсказания:")
        print(f"Вероятность диабета: {result['probability']:.4f}")
        print(f"Класс: {result['class']}")
        print(f"Метка: {result['prediction']}")
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
