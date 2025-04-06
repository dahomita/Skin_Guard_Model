from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import uvicorn
import io

app = FastAPI(title="Skin Cancer Detection API")

# Load the trained model
model = load_model('skin_cancer_detection_model.h5')

# Define the size of the input images
img_size = (224, 224)

# HTML content for the test interface
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Skin Cancer API Test Interface</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        .container {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .preview {
            margin-top: 20px;
            display: none;
        }
        .preview img {
            max-width: 100%;
            max-height: 300px;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }
        .result {
            margin-top: 20px;
            display: none;
            padding: 15px;
            border-radius: 5px;
        }
        .loading {
            display: none;
            margin-top: 20px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        input[type="file"] {
            margin: 10px 0;
        }
        .api-url {
            padding: 10px;
            width: 100%;
            box-sizing: border-box;
            margin-bottom: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .cancer {
            background-color: #ffebee;
            border: 1px solid #f44336;
        }
        .non-cancer {
            background-color: #e8f5e9;
            border: 1px solid #4caf50;
        }
        .note {
            background-color: #e1f5fe;
            padding: 10px;
            border-radius: 4px;
            margin-top: 20px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>Skin Cancer Detection API Test</h1>
    
    <div class="container">
        <h2>Upload a skin image for analysis</h2>
        <form id="upload-form">
            <input type="file" id="image-input" accept="image/*" required>
            <button type="submit">Analyze Image</button>
        </form>
        
        <div class="loading" id="loading">
            Analyzing image... Please wait.
        </div>
        
        <div class="preview" id="preview">
            <h3>Image Preview</h3>
            <img id="preview-image" src="" alt="Preview">
        </div>
        
        <div class="result" id="result">
            <h3>Analysis Result</h3>
            <p id="result-text"></p>
            <p>Probability: <span id="probability"></span></p>
        </div>
    </div>

    <script>
        document.getElementById('image-input').addEventListener('change', function(event) {
            const preview = document.getElementById('preview');
            const previewImg = document.getElementById('preview-image');
            const file = event.target.files[0];
            
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImg.src = e.target.result;
                    preview.style.display = 'block';
                }
                reader.readAsDataURL(file);
            }
        });
        
        document.getElementById('upload-form').addEventListener('submit', async function(event) {
            event.preventDefault();
            
            const fileInput = document.getElementById('image-input');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select an image file');
                return;
            }
            
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const resultText = document.getElementById('result-text');
            const probability = document.getElementById('probability');
            
            // Show loading, hide previous results
            loading.style.display = 'block';
            result.style.display = 'none';
            
            // Create form data and append the file
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error ${response.status}`);
                }
                
                const data = await response.json();
                
                // Display the results
                resultText.textContent = `Prediction: ${data.pred_label}`;
                probability.textContent = `${(data.pred_prob * 100).toFixed(2)}%`;
                
                // Style based on result
                result.className = 'result';
                if (data.pred_label === 'Cancer') {
                    result.classList.add('cancer');
                } else {
                    result.classList.add('non-cancer');
                }
                
                // Show results
                result.style.display = 'block';
            } catch (error) {
                alert('Error: ' + error.message);
                console.error('Error:', error);
            } finally {
                loading.style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""

# Define a function to preprocess the input image
def preprocess_image(img):
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

@app.get("/", response_class=HTMLResponse)
async def get_html():
    """Serve the HTML test interface."""
    return HTML

@app.post("/predict", response_class=JSONResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict if the uploaded skin image contains cancer.
    Returns prediction label and probability.
    """
    # Read the image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocess the image
    processed_img = preprocess_image(img)
    
    # Make prediction
    pred = model.predict(processed_img)
    pred_prob = float(pred[0][0])
    pred_label = "Cancer" if pred_prob > 0.5 else "Not Cancer"
    
    # Return prediction as JSON
    return {
        "pred_label": pred_label,
        "pred_prob": round(pred_prob, 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 