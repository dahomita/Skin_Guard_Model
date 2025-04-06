from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('skin_cancer_detection_model.h5')

# Define the size of the input images
img_size = (224, 224)

# Define a function to preprocess the input image
def preprocess_image(img):
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Define an endpoint to handle image uploads
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file:
        # Read the image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        # Preprocess the image
        img = preprocess_image(img)
        # Make a prediction
        pred = model.predict(img)
        pred_label = 'Cancer' if pred[0][0] > 0.5 else 'Not Cancer'
        pred_prob = float(pred[0][0])
        # Return the prediction result
        return jsonify({'prediction': pred_label, 'probability': pred_prob})

    return jsonify({'error': 'Invalid image'}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
