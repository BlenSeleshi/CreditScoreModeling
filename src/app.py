from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained Random Forest model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest_model.pkl')
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json()

    # Assume data is a list of features for the prediction
    features = np.array(data['features']).reshape(1, -1)

    # Predict using the loaded model
    prediction = model.predict(features)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
