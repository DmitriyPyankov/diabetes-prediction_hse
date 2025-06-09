<!-- README.md -->
# Diabetes Prediction ML Project  
**Проект для бинарной классификации наличия диабета у пациента** с использованием машинного обучения и веб-интерфейса на Gradio.  

## Быстрый старт  
1. **Клонирование репозитория**  
```bash
git clone https://github.com/DmitriyPyankov/diabetes-prediction_hse.git
cd diabetes-prediction_hse
```

2. **Установка зависимостей**  
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

3. **Обучение модели**  
```bash
python src/train.py
```

Модель сохраняется в models/ с метриками.  

4. **Запуск приложения**  
```bash
python src
```

Интерфейс будет доступен по http://localhost:7860


# Docker-развёртывание
```bash
docker build -t diabetes-app .
docker run -p 7860:7860 diabetes-app
```
Приложение будет доступно по http://localhost:7860


## Структура проекта  
 .  
├── data/               # Исходные данные (diabetes.csv)  
├── models/             # Сохранённые модели и метрики   
├── notebooks/          # Jupyter-ноутбуки для анализа и экспериментов   
├── src/  
│   ├── train.py        # Скрипт обучения модели  
│   ├── predict.py      # Функции загрузки модели и предсказания  
│   ├── app.py          # Gradio-интерфейс приложения  
│   └── __main__.py     # Запуск приложения через пакет src  
├── requirements.txt    # Python-зависимости  
├── Dockerfile          # Конфигурация Docker-образа  
└── README.md           # Документация проекта  
  
**Описание ключевых модулей**  
train.py — загружает данные, обучает CatBoostClassifier, оценивает качество, сохраняет модель и метрики.  
  
predict.py — загружает последнюю модель, делает предсказания на новых данных.  
  
app.py — реализует веб-интерфейс с помощью Gradio, принимает пользовательский ввод и выводит результат.  
  
main.py — позволяет запускать приложение командой python -m src.  