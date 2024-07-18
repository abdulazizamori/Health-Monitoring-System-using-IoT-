from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("rf_model.pkl")

# Load the LabelEncoder
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Load the SelectKBest object
    selector = joblib.load("select_k_best.pkl")

    # Transform the input features using SelectKBest
    X_selected = selector.transform(data['features'])

    # Make prediction
    prediction = model.predict(X_selected)

    # Decode the encoded labels
    predicted_labels = label_encoder.inverse_transform(prediction)

    # Return the prediction
    return jsonify({'prediction': f'{predicted_labels.tolist()}'})


if __name__ == '__main__':
    app.run(port=8000, debug=True)