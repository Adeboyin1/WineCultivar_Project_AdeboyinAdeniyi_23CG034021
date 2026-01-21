# Wine Cultivar Origin Prediction System - Flask Application
# Alternative Part B: Web GUI Application using Flask

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model components
model = joblib.load('model/wine_cultivar_model.pkl')
scaler = joblib.load('model/scaler.pkl')
feature_names = joblib.load('model/feature_names.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        data = request.get_json()
        
        # Extract features
        input_data = np.array([[
            float(data['alcohol']),
            float(data['flavanoids']),
            float(data['color_intensity']),
            float(data['od280_od315']),
            float(data['proline']),
            float(data['total_phenols'])
        ]])
        
        # Create DataFrame with feature names
        input_df = pd.DataFrame(input_data, columns=feature_names)
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Cultivar names
        cultivar_names = {
            0: "Cultivar 0 (Class 1)",
            1: "Cultivar 1 (Class 2)",
            2: "Cultivar 2 (Class 3)"
        }
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'cultivar_name': cultivar_names[prediction],
            'confidence': float(prediction_proba[prediction] * 100),
            'probabilities': {
                cultivar_names[i]: float(prob * 100) 
                for i, prob in enumerate(prediction_proba)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)