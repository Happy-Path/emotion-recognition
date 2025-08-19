import requests
import json
import os
from pathlib import Path

# API configuration
API_BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print("🏥 Health Check:")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print("\n🏠 Root Endpoint:")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

def test_emotion_prediction(image_path, endpoint="predict-emotion"):
    """Test emotion prediction with an image"""
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False

    try:
        with open(image_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            response = requests.post(f"{API_BASE_URL}/{endpoint}/", files=files)

        print(f"\n🎭 Emotion Prediction ({endpoint}):")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Emotion prediction failed: {e}")
        return False

def create_test_image():
    """Create a simple test image using OpenCV"""
    try:
        import cv2
        import numpy as np

        # Create a simple test image (gray square)
        img = np.ones((48, 48, 3), dtype=np.uint8) * 128

        # Add some simple features to make it look like a face
        cv2.circle(img, (15, 20), 3, (0, 0, 0), -1)  # Left eye
        cv2.circle(img, (33, 20), 3, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(img, (24, 35), (8, 4), 0, 0, 180, (0, 0, 0), 2)  # Mouth

        test_image_path = "test_image.jpg"
        cv2.imwrite(test_image_path, img)
        print(f"✅ Created test image: {test_image_path}")
        return test_image_path

    except ImportError:
        print("❌ OpenCV not available, cannot create test image")
        return None
    except Exception as e:
        print(f"❌ Failed to create test image: {e}")
        return None

def main():
    print("🧪 Testing Emotion Recognition API")
    print("=" * 50)

    # Test 1: Health check
    health_ok = test_health_endpoint()

    # Test 2: Root endpoint
    root_ok = test_root_endpoint()

    # Test 3: Create and test with a simple image
    test_image_path = create_test_image()

    prediction_ok = False
    simple_prediction_ok = False

    if test_image_path:
        # Test 4: Full emotion prediction with face detection
        prediction_ok = test_emotion_prediction(test_image_path, "predict-emotion")

        # Test 5: Simple emotion prediction
        simple_prediction_ok = test_emotion_prediction(test_image_path, "predict-emotion-simple")

        # Clean up test image
        try:
            os.remove(test_image_path)
            print(f"\n🗑️ Cleaned up test image: {test_image_path}")
        except:
            pass

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"Health Check: {'✅' if health_ok else '❌'}")
    print(f"Root Endpoint: {'✅' if root_ok else '❌'}")
    print(f"Emotion Prediction (with face detection): {'✅' if prediction_ok else '❌'}")
    print(f"Simple Emotion Prediction: {'✅' if simple_prediction_ok else '❌'}")

    all_tests_passed = all([health_ok, root_ok, prediction_ok, simple_prediction_ok])
    print(f"\nOverall: {'🎉 All tests passed!' if all_tests_passed else '⚠️ Some tests failed'}")

    if not all_tests_passed:
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure the FastAPI server is running: python app.py")
        print("2. Check if the model file exists: model/emotion_model.keras")
        print("3. Verify all dependencies are installed: pip install -r requirements.txt")
        print("4. Make sure the server is accessible at http://localhost:8000")

if __name__ == "__main__":
    main()