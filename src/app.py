# app.py
def main():
    import gradio as gr
    import numpy as np
    import pandas as pd
    from predict import load_latest_model, predict_diabetes

    # Загрузка последней модели
    model, _ = load_latest_model()

    def gradio_predict(
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree_function, age
    ):
        input_data = {
            'pregnancies': pregnancies,
            'glucose': glucose,
            'blood_pressure': blood_pressure,
            'skin_thickness': skin_thickness,
            'insulin': insulin,
            'bmi': bmi,
            'diabetes_pedigree_function': diabetes_pedigree_function,
            'age': age
        }
        
        result = predict_diabetes(model, input_data)
        probability = result["probability"]
        label = "Диабет" if result["prediction"] == 1 else "Здоров"
        return f"{label} (вероятность: {probability:.2%})"

    # Входные поля
    inputs = [
        gr.Number(label="Беременности (например: 2)"),
        gr.Number(label="Глюкоза (mg/dl, например: 120)"),
        gr.Number(label="Давление (mm Hg, например: 70)"),
        gr.Number(label="Толщина кожи (mm, например: 20)"),
        gr.Number(label="Инсулин (mu U/ml, например: 85)"),
        gr.Number(label="BMI (например: 24.5)"),
        gr.Number(label="Diabetes Pedigree (например: 0.35)"),
        gr.Number(label="Возраст (например: 35)"),
    ]

    # Интерфейс
    gr.Interface(
        fn=gradio_predict,
        inputs=inputs,
        outputs="text",
        title="Предсказание диабета",
        description="Введите параметры пациента для оценки вероятности диабета"
    ).launch(server_port=7860, share=True)

if __name__ == "__main__":
    main()