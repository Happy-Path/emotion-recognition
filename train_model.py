import pandas as pd
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("fer2013.csv")

# Preprocess
pixels = df['pixels'].tolist()
faces = np.array([np.fromstring(pixel, sep=' ').reshape(48, 48) for pixel in pixels])
faces = faces.astype('float32') / 255.0
faces = np.expand_dims(faces, -1)

emotions = to_categorical(df['emotion'], num_classes=7)

# Split data
X_train, X_val, y_train, y_val = train_test_split(faces, emotions, test_size=0.2, random_state=42)

# CNN Model
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ⏹ EarlyStopping Callback
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=100,  # Max possible epochs
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=2
)

# Evaluate model on test data
test_loss, test_accuracy = model.evaluate(X_val, y_val, verbose=2)
print(f"📊 Test Accuracy: {test_accuracy * 100:.2f}%")

# Save model
if not os.path.exists("model"):
    os.makedirs("model")

model.save("model/emotion_model.keras")
print("✅ Model saved to model/emotion_model.keras")
