# for API flask
from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('imputer.pkl', 'rb') as imputer_file:
    imputer = pickle.load(imputer_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        features = pd.DataFrame([data['features']], columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'])
        

        features[['total_bedrooms']] = imputer.transform(features[['total_bedrooms']])
        
        ocean_proximity_encoded = encoder.transform(features[['ocean_proximity']])
        ocean_proximity_encoded_df = pd.DataFrame(ocean_proximity_encoded.toarray(), columns=encoder.get_feature_names_out(['ocean_proximity']))
        
        features = features.drop('ocean_proximity', axis=1)
        
        features_final = pd.concat([features, ocean_proximity_encoded_df.reset_index(drop=True)], axis=1)

        
        features_scaled = scaler.transform(features_final)
        
        prediction = model.predict(features_scaled)
        
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
@app.route('/')
def serve_html():
    return send_from_directory('', 'index.html')


if __name__ == '__main__':
    app.run(debug=True)



