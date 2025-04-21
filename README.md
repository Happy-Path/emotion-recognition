# Clone the repository
git clone https://github.com/your-username/emotion-recognition.git
cd emotion-recognition

# Set up and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# (Manually) download fer2013.csv from https://www.kaggle.com/datasets/msambare/fer2013
# and place it in the root project directory

# Train the model
python train_model.py

# Run real-time emotion detection
python main.py

# Press 'q' to quit webcam window
