from flask import Flask, render_template, request, jsonify
from src.utils import load_object
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load artifacts at startup
model = load_object('artifacts/model.pkl')
preprocessor = load_object('artifacts/preprocessor.pkl')

# Feature list (order matters)
REQUIRED_FEATURES = [
    'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
    'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding',
    'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
    'Departure Delay in Minutes', 'Arrival Delay in Minutes'
]

@app.route('/')
def home():
    return render_template('index.html', features=REQUIRED_FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Preprocess
        X_transformed = preprocessor.transform(input_df)
        
        # Predict
        proba = model.predict_proba(X_transformed)[0, 1]
        prediction = 'satisfied' if proba >= 0.6 else 'neutral or dissatisfied'
        
        # Risk level for service recovery
        if proba >= 0.7:
            risk_level = 'low'
            action = 'No action needed'
        elif proba >= 0.5:
            risk_level = 'medium'
            action = 'Monitor'
        else:
            risk_level = 'high'
            action = 'Service recovery recommended - contact passenger'
        
        result = {
            'prediction': prediction,
            'probability_satisfied': round(float(proba), 3),
            'risk_level': risk_level,
            'recommended_action': action
        }
        
        if request.is_json:
            return jsonify(result)
        else:
            return render_template('result.html', result=result, input_data=data)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model': 'Logistic Regression loaded'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
