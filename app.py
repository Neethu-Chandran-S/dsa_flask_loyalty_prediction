from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained models and encoders
scaler = joblib.load('std_scalar (1).pkl')
encoder = joblib.load('l_encoder (1).pkl')

lr_model = joblib.load('lr_newmodel.pkl')


# Feature list for proper input mapping
selected_features = ['Revenue','Total Spent','Satisfaction Score']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON data
    data = request.json
    try:
        # Convert to DataFrame for processing
        input_data = pd.DataFrame([data])

        # Scale numerical features
        input_data[['Revenue', 'Total Spent', 'Satisfaction Score']] = scaler.transform(
            input_data[['Revenue', 'Total Spent', 'Revenue']]
        )

     
        
       

        # Select RFE-selected features
        input_data = input_data[selected_features]

        # Make prediction
        prediction = lr_model.predict(input_data)
      

        return jsonify({'Loyalty Score': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__== '_main_':
    app.run(debug=True)