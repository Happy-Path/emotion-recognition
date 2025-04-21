import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Path to the model file
model_path = "model/emotion_model.keras"

# Check if model exists
if not os.path.exists(model_path):
    print("❌ Model file not found at:", model_path)
    exit()

# Load the model
model = load_model(model_path)
print("✅ Model loaded successfully.")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize webcam
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try CAP_DSHOW to fix webcam access issues

if not cap.isOpened():
    print("❌ Error: Could not access the webcam.")
    exit()
else:
    print("📸 Webcam is working!")

# Load face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Real-time emotion detection loop
while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Iterate through all detected faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi_gray, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)  # Preprocess the face image

        # Predict the emotion
        prediction = model.predict(roi, verbose=0)[0]
        label = emotion_labels[np.argmax(prediction)]

        # Draw bounding box and emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame with emotion labels
    cv2.imshow('Real-time Emotion Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
