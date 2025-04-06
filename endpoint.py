from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()
model = load_model('skin_cancer_detection_model.h5')
img_size = (224, 224)

def preprocess_image(img):
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        print("[INFO] Received image")
        contents = await image.read()

        # Decode the image
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), 1)
        if img is None:
            print("[ERROR] Failed to decode image")
            return JSONResponse(content={"error": "Invalid image"}, status_code=400)

        # Preprocess the image
        print("[INFO] Preprocessing...")
        img = preprocess_image(img)

        # Predict
        print("[INFO] Predicting...")
        pred = model.predict(img)
        pred_label = 'Cancer' if pred[0][0] > 0.5 else 'Not Cancer'
        pred_prob = float(pred[0][0])

        print("[INFO] Prediction successful")
        return {"prediction": pred_label, "probability": round(pred_prob, 4)}

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
