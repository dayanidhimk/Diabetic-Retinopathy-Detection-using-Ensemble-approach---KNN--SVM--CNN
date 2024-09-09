from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.ensemble import VotingClassifier

app = Flask(__name__)

# Load models
knn_model = joblib.load('knn_model.joblib')
svm_model = joblib.load('svm_model.joblib')
cnn_model = tf.keras.models.load_model('cnn_model.h5')
scaler = joblib.load('scaler.joblib')

# Preprocessing function
def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img)
    return img_array

# Map predictions to DR levels
def map_prediction(prediction):
    dr_levels = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferate DR"]
    return dr_levels[int(prediction)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    img_array = preprocess_image(file)
    img_flatten = img_array.flatten()

    # KNN and SVM prediction
    img_flatten_scaled = scaler.transform([img_flatten])
    knn_prediction = knn_model.predict(img_flatten_scaled)[0]
    svm_prediction = svm_model.predict(img_flatten_scaled)[0]

    # CNN prediction
    img_array_normalized = img_array / 255.0
    img_array_expanded = np.expand_dims(img_array_normalized, axis=0)
    cnn_prediction = cnn_model.predict(img_array_expanded)[0][0]
    cnn_prediction_rounded = round(cnn_prediction)

    # Voting Classifier
    predictions = [knn_prediction, svm_prediction, cnn_prediction_rounded]
    final_prediction = max(set(predictions), key=predictions.count)

    return jsonify({
        'knn_prediction': map_prediction(knn_prediction),
        'svm_prediction': map_prediction(svm_prediction),
        'cnn_prediction': map_prediction(cnn_prediction_rounded),
        'ensemble_prediction': map_prediction(final_prediction)
    })

if __name__ == '__main__':
    app.run(debug=True)
