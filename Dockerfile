# Используем официальный Python-образ
FROM python:3.11-slim

# Обновление pip и установка зависимостей системы
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории внутри контейнера
WORKDIR /app

# Копируем зависимости и проект в контейнер
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект внутрь контейнера
COPY . .

# Указываем порт, который будет открыт
EXPOSE 7860

# Команда по умолчанию — запуск Gradio-приложения
CMD ["python", "src/app.py"]