# Emotion Recognition Using Deep Learning

This project is a real-time emotion recognition system using a Convolutional Neural Network (CNN) for facial expression recognition. The model uses the FER-2013 dataset and leverages OpenCV for real-time webcam emotion detection.

## Project Overview

The model is trained to recognize seven basic emotions from facial expressions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

The project includes:
- **Train Model**: A Python script that trains the model on the FER-2013 dataset.
- **Detect Emotion**: A Python script for real-time emotion detection using a webcam.
- **Main Script**: A Python script for integrating the webcam feed with the emotion recognition model.

## Prerequisites

Before you run the project, make sure you have the following installed:

- Python 3.6 or higher
- OpenCV
- TensorFlow/Keras
- NumPy
- pandas
- scikit-learn

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Happy-Path/emotion-recognition

2. Create and activate a virtual environment:

    python -m venv venv

On Windows:

    .\venv\Scripts\activate

On macOS/Linux:

    source venv/bin/activate

3. Install the dependencies:

    pip install -r requirements.txt

4. Download the FER-2013 dataset (fer2013.csv) and place it in the project directory. The dataset can be obtained from [this link](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition).

## Files

    train_model.py: Script to train the emotion recognition model.
    main.py: Combined script for webcam emotion detection.
    fer2013.csv: The FER-2013 dataset containing images and labels (download separately).

## How to Train the Model

    First, train the model by running:

    python train_model.py

    This script will train a Convolutional Neural Network (CNN) on the FER-2013 dataset and save the trained model as model/emotion_model.keras.

## Real-time Emotion Detection

    To start real-time emotion detection using your webcam, run:

    python main.py

    This will open a webcam window showing live emotion predictions on the detected faces.

## Exit the program

    Press 'q' to exit the webcam window.

## License

    This project is licensed under the MIT License - see the LICENSE file for details.