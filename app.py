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
    
    # Check if model has feature names
    if hasattr(model, 'feature_names_in_'):
        logger.info(f"Model expects features: {model.feature_names_in_.tolist()}")
    if hasattr(model, 'classes_'):
        logger.info(f"Model classes: {model.classes_}")
        
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Expected feature order
FEATURES = [
    'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'BMI',
    'Previous Complications', 'Preexisting Diabetes',
    'Gestational Diabetes', 'Mental Health', 'HeartRate'
]

@app.route('/')
def home():
    return "Maternal Health Prediction API is running!"

@app.route('/debug_predict', methods=['POST'])
def debug_predict():
    """Debug endpoint to see what's happening inside the model"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        data = request.get_json()
        logger.info(f"📊 DEBUG - Received data: {data}")

        # Convert to DataFrame and ensure correct data types
        input_df = pd.DataFrame([data])
        input_df = input_df.astype({
            'Age': float, 'Systolic BP': float, 'Diastolic': float, 
            'BS': float, 'Body Temp': float, 'BMI': float,
            'Previous Complications': int, 'Preexisting Diabetes': int,
            'Gestational Diabetes': int, 'Mental Health': int, 'Heart Rate': float
        })

        # Reorder columns to match training data
        input_df = input_df.reindex(columns=FEATURES)
        
        logger.info(f"📊 DEBUG - Processed DataFrame:")
        logger.info(f"Columns: {input_df.columns.tolist()}")
        logger.info(f"Data types: {input_df.dtypes.to_dict()}")
        logger.info(f"Values: {input_df.values.tolist()}")

        # Get raw prediction
        prediction = model.predict(input_df)[0]
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df)[0]
            logger.info(f"📊 DEBUG - Prediction probabilities: {probabilities}")
        
        # Try to get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(FEATURES, model.feature_importances_))
            logger.info(f"📊 DEBUG - Feature importance: {feature_importance}")

        logger.info(f"📊 DEBUG - Final prediction: {prediction}")

        return jsonify({
            "risk_level": str(prediction),
            "probabilities": probabilities.tolist() if probabilities is not None else None,
            "feature_importance": feature_importance,
            "processed_data": input_df.iloc[0].to_dict(),
            "debug_info": {
                "model_type": str(type(model)),
                "features_received": list(data.keys()),
                "features_processed": FEATURES
            }
        })

    except Exception as e:
        logger.error(f"📊 DEBUG - Prediction error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        data = request.get_json()

        # Convert to DataFrame and ensure correct data types
        input_df = pd.DataFrame([data])
        input_df = input_df.astype({
            'Age': float, 'Systolic BP': float, 'Diastolic': float, 
            'BS': float, 'Body Temp': float, 'BMI': float,
            'Previous Complications': int, 'Preexisting Diabetes': int,
            'Gestational Diabetes': int, 'Mental Health': int, 'Heart Rate': float
        })

        # Reorder columns to match training data
        input_df = input_df.reindex(columns=FEATURES)

        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df)[0]
        else:
            probabilities = None

        return jsonify({
            "risk_level": str(prediction),
            "probabilities": probabilities.tolist() if probabilities is not None else None
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/test_specific', methods=['POST'])
def test_specific():
    """Test specific case that should be low risk"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Your specific test case that should be low risk
        test_case = {
            'Age': 22.0, 
            'Systolic BP': 110.0, 
            'Diastolic': 70.0, 
            'BS': 7.1, 
            'Body Temp': 98.0, 
            'BMI': 20.4,
            'Previous Complications': 0, 
            'Preexisting Diabetes': 0,
            'Gestational Diabetes': 0, 
            'Mental Health': 0, 
            'Heart Rate': 74.0
        }
        
        input_df = pd.DataFrame([test_case])
        input_df = input_df.reindex(columns=FEATURES)
        
        logger.info(f"🔍 SPECIFIC TEST - Input data: {test_case}")
        logger.info(f"🔍 SPECIFIC TEST - Processed data: {input_df.values.tolist()}")
        
        prediction = model.predict(input_df)[0]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df)[0]
            logger.info(f"🔍 SPECIFIC TEST - Probabilities: {probabilities}")
        else:
            probabilities = None
            
        logger.info(f"🔍 SPECIFIC TEST - Prediction: {prediction}")
        
        return jsonify({
            "test_case": "Age=22, BP=110/70, BS=7.1, BMI=20.4 (Should be Low Risk)",
            "prediction": int(prediction),
            "probabilities": probabilities.tolist() if probabilities is not None else None,
            "expected": "Low Risk (0)",
            "actual": f"{'✅ MATCH' if prediction == 0 else '❌ MISMATCH'}"
        })
        
    except Exception as e:
        logger.error(f"Specific test error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

