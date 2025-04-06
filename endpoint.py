from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # <- Import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# ✅ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

        img = cv2.imdecode(np.frombuffer(contents, np.uint8), 1)
        if img is None:
            return JSONResponse(content={"error": "Invalid image"}, status_code=400)

        img = preprocess_image(img)

        pred = model.predict(img)
        pred_label = 'Cancer' if pred[0][0] > 0.5 else 'Not Cancer'
        pred_prob = float(pred[0][0])

        return {"prediction": pred_label, "probability": round(pred_prob, 4)}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
