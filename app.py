from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained XGBoost pipeline
model = joblib.load("xgboost_pipeline.pkl")

@app.route('/')
def home():
    return "Maternal Health Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Convert JSON to DataFrame (expect keys match your features)
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)[0]

        return jsonify({"risk_level": str(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
