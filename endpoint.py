from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Load the trained model
model = load_model('skin_cancer_detection_model.h5')

# ✅ Define image size
img_size = (224, 224)

# ✅ Preprocessing function
def preprocess_image(img):
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# ✅ Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file:
        try:
            # Read and decode image
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
            # Preprocess
            img = preprocess_image(img)
            # Predict
            pred = model.predict(img)
            pred_label = 'Cancer' if pred[0][0] > 0.5 else 'Not Cancer'
            pred_prob = float(pred[0][0])
            # Respond
            return jsonify({'prediction': pred_label, 'probability': pred_prob})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid image'}), 400

# ⚠️ Do NOT include `app.run()` — gunicorn will launch the app
