from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the trained XGBoost pipeline
try:
    model = joblib.load("xgboost_pipeline.pkl")
    logger.info("Model loaded successfully")
    logger.info(f"Model type: {type(model)}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Expected feature order
FEATURES = [
    'Age', 'Systolic BP', 'Diastolic', 'BS', 'Body Temp', 'BMI',
    'Previous Complications', 'Preexisting Diabetes',
    'Gestational Diabetes', 'Mental Health', 'Heart Rate'
]

@app.route('/')
def home():
    return "Maternal Health Prediction API is running!"

@app.route('/test', methods=['GET'])
def test_model():
    """Test endpoint to verify model is working with sample data"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Test cases that should give different predictions
        test_cases = [
            # Low risk case (should predict 0)
            {
                'Age': 25.0, 'Systolic BP': 110.0, 'Diastolic': 70.0, 
                'BS': 6.0, 'Body Temp': 98.6, 'BMI': 22.0,
                'Previous Complications': 0, 'Preexisting Diabetes': 0,
                'Gestational Diabetes': 0, 'Mental Health': 0, 'Heart Rate': 72.0
            },
            # Medium risk case (should predict 1)
            {
                'Age': 32.0, 'Systolic BP': 130.0, 'Diastolic': 85.0, 
                'BS': 8.5, 'Body Temp': 99.0, 'BMI': 28.0,
                'Previous Complications': 1, 'Preexisting Diabetes': 0,
                'Gestational Diabetes': 1, 'Mental Health': 0, 'Heart Rate': 85.0
            },
            # High risk case (should predict 2)
            {
                'Age': 38.0, 'Systolic BP': 150.0, 'Diastolic': 95.0, 
                'BS': 12.0, 'Body Temp': 100.2, 'BMI': 35.0,
                'Previous Complications': 1, 'Preexisting Diabetes': 1,
                'Gestational Diabetes': 1, 'Mental Health': 1, 'Heart Rate': 95.0
            }
        ]
        
        results = []
        for i, test_data in enumerate(test_cases):
            input_df = pd.DataFrame([test_data])
            input_df = input_df.reindex(columns=FEATURES)
            
            logger.info(f"Test {i+1} data:")
            logger.info(f"Data types: {input_df.dtypes.to_dict()}")
            logger.info(f"Values: {input_df.values.tolist()}")
            
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
            
            result = {
                'test_case': i+1,
                'prediction': int(prediction),
                'probabilities': prediction_proba.tolist() if prediction_proba is not None else None,
                'input_data': test_data
            }
            results.append(result)
        
        return jsonify({"test_results": results})
        
    except Exception as e:
        logger.error(f"Test error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")

        # Validate all required fields are present
        missing_fields = [field for field in FEATURES if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {missing_fields}"}), 400

        # Convert to DataFrame and ensure correct data types
        input_df = pd.DataFrame([data])
        
        # Convert to proper data types
        input_df = input_df.astype({
            'Age': float, 'Systolic BP': float, 'Diastolic': float, 
            'BS': float, 'Body Temp': float, 'BMI': float,
            'Previous Complications': int, 'Preexisting Diabetes': int,
            'Gestational Diabetes': int, 'Mental Health': int, 'Heart Rate': float
        })

        # Reorder columns to match training data
        input_df = input_df.reindex(columns=FEATURES)
        
        logger.info(f"Processed DataFrame dtypes: {input_df.dtypes.to_dict()}")
        logger.info(f"Processed DataFrame values: {input_df.values.tolist()}")

        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df)[0]
            logger.info(f"Prediction probabilities: {probabilities}")
        else:
            probabilities = None

        logger.info(f"Final prediction: {prediction}")

        return jsonify({
            "risk_level": str(prediction),
            "probabilities": probabilities.tolist() if probabilities is not None else None
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/features', methods=['GET'])
def get_features():
    """Endpoint to show expected features and their types"""
    feature_info = {
        'Age': {'type': 'float', 'description': 'Age in years'},
        'Systolic BP': {'type': 'float', 'description': 'Systolic blood pressure in mmHg'},
        'Diastolic': {'type': 'float', 'description': 'Diastolic blood pressure in mmHg'},
        'BS': {'type': 'float', 'description': 'Blood sugar level'},
        'Body Temp': {'type': 'float', 'description': 'Body temperature in °F'},
        'BMI': {'type': 'float', 'description': 'Body Mass Index'},
        'Previous Complications': {'type': 'int', 'description': '0=No, 1=Yes'},
        'Preexisting Diabetes': {'type': 'int', 'description': '0=No, 1=Yes'},
        'Gestational Diabetes': {'type': 'int', 'description': '0=No, 1=Yes'},
        'Mental Health': {'type': 'int', 'description': '0=No, 1=Yes'},
        'Heart Rate': {'type': 'float', 'description': 'Heart rate in bpm'}
    }
    return jsonify({"expected_features": feature_info})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
