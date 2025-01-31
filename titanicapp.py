from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Initializing the Flask App
app = Flask(__name__)

# Loading the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Expected Features for prediction
expected_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S']

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate input format
        if not data or 'features' not in data:
            return jsonify({'error': 'Invalid Input: Expected JSON with "features" key'}), 400

        features = data['features']
        if not isinstance(features, list) or not all(isinstance(row, dict) for row in features):
            return jsonify({'error': f'Features should be a list of dictionaries with keys: {expected_features}'}), 400

        # Convert the input to a DataFrame
        input_df = pd.DataFrame(features)

        # Ensure all the expected features are present
        for feature in expected_features:
            if feature not in input_df.columns:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

        # Convert numeric features to float and handle the missing values
        numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        for feature in numeric_features:
            input_df[feature] = pd.to_numeric(input_df[feature], errors='coerce')

            if input_df[feature].isnull().any():
                return jsonify({'error': f'Invalid or missing numeric value in {feature}'}), 400

        # Ensure categorical features are of correct type
        input_df[['male', 'Q', 'S']] = input_df[['male', 'Q', 'S']].astype(int)

        # Convert all the input data to model-compatible format
        input_data = input_df[expected_features].values.tolist()

        # Make predictions
        predictions = model.predict(input_data)

        # Format response
        response = {
            'predictions': predictions.tolist()
        }  

        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
