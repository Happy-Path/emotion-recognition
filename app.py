from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os
from typing import List, Dict
import uvicorn

app = FastAPI(title="Emotion Recognition API", version="1.0.0")

# Configure CORS - Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "*"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
face_classifier = None

@app.on_event("startup")
async def load_model_and_classifier():
    global model, face_classifier

    # Load emotion model
    model_path = "model/emotion_model.keras"
    if not os.path.exists(model_path):
        raise Exception(f"Model file not found at {model_path}")

    model = load_model(model_path)
    print("✅ Emotion model loaded successfully")

    # Load face classifier
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("✅ Face classifier loaded successfully")

def preprocess_face(face_img):
    """Preprocess face image for emotion prediction"""
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype("float32") / 255.0
    face_img = img_to_array(face_img)
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

@app.post("/predict-emotion/")
async def predict_emotion(file: UploadFile = File(...)):
    """Predict emotion from uploaded image with face detection"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_classifier.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return {
                "success": False,
                "message": "No faces detected in the image",
                "faces_count": 0,
                "predictions": []
            }

        predictions = []

        # Process each detected face
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]

            # Preprocess face
            processed_face = preprocess_face(face_roi)

            # Predict emotion
            emotion_pred = model.predict(processed_face, verbose=0)[0]
            emotion_idx = int(np.argmax(emotion_pred))
            emotion_label = emotions[emotion_idx]
            confidence = float(emotion_pred[emotion_idx])

            # Create prediction result
            face_prediction = {
                "face_id": i + 1,
                "emotion": emotion_label,
                "confidence": round(confidence * 100, 2),
                "coordinates": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                },
                "all_probabilities": {
                    emotions[j]: round(float(emotion_pred[j]) * 100, 2)
                    for j in range(len(emotions))
                }
            }
            predictions.append(face_prediction)

        return {
            "success": True,
            "message": f"Successfully detected {len(faces)} face(s)",
            "faces_count": len(faces),
            "predictions": predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict-emotion-simple/")
async def predict_emotion_simple(file: UploadFile = File(...)):
    """Simple emotion prediction without face detection (for pre-cropped faces)"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Preprocess image
        processed_img = preprocess_face(img)

        # Predict emotion
        emotion_pred = model.predict(processed_img, verbose=0)[0]
        emotion_idx = int(np.argmax(emotion_pred))
        emotion_label = emotions[emotion_idx]
        confidence = float(emotion_pred[emotion_idx])

        return {
            "success": True,
            "emotion": emotion_label,
            "confidence": round(confidence * 100, 2),
            "all_probabilities": {
                emotions[i]: round(float(emotion_pred[i]) * 100, 2)
                for i in range(len(emotions))
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Emotion Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "/predict-emotion/": "Upload image for emotion detection with face detection",
            "/predict-emotion-simple/": "Upload pre-cropped face image for emotion detection",
            "/health": "Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "face_classifier_loaded": face_classifier is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)