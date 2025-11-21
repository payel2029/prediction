import pandas as pd
import joblib

# Load your trained pipeline
pipeline = joblib.load("xgboost_pipeline.pkl")  # or maternal_pipeline.pkl

def predict_maternal_risk(age, systolic_bp, diastolic, bs, body_temp, bmi,
                          previous_complications, preexisting_diabetes,
                          gestational_diabetes, mental_health, heart_rate):
    """
    Takes user-friendly clinical inputs and returns predicted risk class and probabilities.
    """
    # Create DataFrame from user input
    df_input = pd.DataFrame([{
        'Age': age,
        'Systolic BP': systolic_bp,
        'Diastolic': diastolic,
        'BS': bs,
        'Body Temp': body_temp,
        'BMI': bmi,
        'Previous Complications': previous_complications,
        'Preexisting Diabetes': preexisting_diabetes,
        'Gestational Diabetes': gestational_diabetes,
        'Mental Health': mental_health,
        'Heart Rate': heart_rate
    }])

    # Predict class
    predicted_class = pipeline.predict(df_input)[0]

    # Predict probabilities
    probabilities = pipeline.predict_proba(df_input)[0]

    # Map numeric class to human-readable risk
    class_mapping = {0: 'Low', 1: 'Mid', 2: 'High'}
    predicted_risk = class_mapping.get(predicted_class, "Unknown")

    return predicted_risk, probabilities

# ============================
# Example usage:
# ============================
pred_risk, probs = predict_maternal_risk(
    age=48,
    systolic_bp=120,
    diastolic=80,
    bs=11,
    body_temp=98,
    bmi=29,
    previous_complications=1,
    preexisting_diabetes=1,
    gestational_diabetes=0,
    mental_health=1,
    heart_rate=0
)

print("Predicted Risk:", pred_risk)
print("Class Probabilities:", probs)
