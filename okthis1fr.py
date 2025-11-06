from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)

# Global variables
model = None
user_mapping = {
    0: {"user_id": "kb", "user_name": "Khoo Ben", "user_st": "2291", "grade": "11", "class": "5/1"},
    1: {"user_id": "kin", "user_name": "Kin Seng", "user_st": "6767", "grade": "11", "class": "5/1"},
    2: {"user_id": "ryk", "user_name": "Ryan Koh", "user_st": "8686", "grade": "11", "class": "5/1"}
}

def load_model():
    """Load the TensorFlow model"""
    global model
    try:
        # First, try to load the actual model file
        if os.path.exists('model001.h5'):
            print("🔄 Loading model from model001.h5...")
            model = tf.keras.models.load_model('model001.h5')
            print("✅ Pre-trained model loaded successfully!")
            return True
        else:
            # Fallback: Create a simple model for testing
            print("⚠️  model001.h5 not found. Creating test model...")
            
            # Create a simple CNN model for 100x100 grayscale images
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(3, activation='softmax')  # 3 classes for kb, kin, ryk
            ])
            
            # Compile the model (weights will be random for testing)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            print("✅ Test model created successfully")
            return True
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
        return False

def preprocess_image(image_data):
    """Preprocess the image for the model"""
    try:
        # Remove data URL prefix if present
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            if ',' in image_data:
                image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        print(f"📐 Original image size: {image.size}, mode: {image.mode}")
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
            print("🔄 Converted to grayscale")
        
        # Resize to 100x100
        image = image.resize((100, 100))
        print(f"📏 Resized to: {image.size}")
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        print(f"🔢 Array shape: {img_array.shape}, range: [{np.min(img_array):.3f}, {np.max(img_array):.3f}]")
        
        # Reshape for model (add batch and channel dimensions)
        input_array = img_array.reshape(1, 100, 100, 1)
        print(f"🎯 Final input shape: {input_array.shape}")
        
        return input_array, True
        
    except Exception as e:
        print(f"❌ Image preprocessing failed: {e}")
        return None, False

@app.route('/process-face', methods=['POST'])
def process_face():
    """
    Main endpoint for face recognition - called by Node-RED
    """
    try:
        print("\n" + "="*50)
        print("🎯 Received face recognition request")
        print("="*50)
        
        # Get JSON data
        data = request.get_json()
        if not data:
            print("❌ No JSON data received")
            return jsonify({
                "success": False, 
                "recognized": False,
                "error": "No JSON data received"
            }), 400
        
        print(f"📦 Request keys: {list(data.keys())}")
        
        # Check for image data
        image_data = data.get('image')
        if not image_data:
            print("❌ No image data found")
            return jsonify({
                "success": False,
                "recognized": False, 
                "error": "No image data found"
            }), 400
        
        print(f"🖼️ Image data received, length: {len(image_data)}")
        
        # Preprocess image
        input_array, success = preprocess_image(image_data)
        if not success:
            return jsonify({
                "success": False,
                "recognized": False,
                "error": "Image preprocessing failed"
            }), 400
        
        # Ensure model is loaded
        if model is None:
            if not load_model():
                return jsonify({
                    "success": False,
                    "recognized": False,
                    "error": "Model not available"
                }), 500
        
        # Make prediction
        print("🤖 Making prediction...")
        prediction = model.predict(input_array, verbose=0)
        print(f"🔍 Raw predictions: {prediction}")
        
        # Get results
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        confidence_percent = round(confidence * 100, 2)
        
        print(f"🎭 Predicted class: {predicted_class}")
        print(f"📊 Confidence: {confidence:.4f} ({confidence_percent}%)")
        
        # Determine recognition (lower threshold for testing)
        recognized = confidence > 0.3  # Lower threshold for testing
        user_info = user_mapping.get(predicted_class, {
            "user_id": "unknown",
            "user_name": "Unknown User", 
            "user_st": "N/A",
            "grade": "N/A",
            "class": "N/A"
        })
        
        status = "✅ Recognized" if recognized else "❌ Not recognized"
        print(f"{status} - User: {user_info['user_name']} (Confidence: {confidence_percent}%)")
        
        # Prepare response in exact format Node-RED expects
        response = {
            "success": True,
            "recognized": recognized,
            "confidence": confidence_percent,
            "user_id": user_info["user_id"],
            "user_name": user_info["user_name"], 
            "user_st": user_info["user_st"],
            "grade": user_info["grade"],
            "class": user_info.get("class", "N/A"),
            "predicted_class": int(predicted_class),
            "message": f"Recognized: {user_info['user_name']}" if recognized else "Face not recognized"
        }
        
        print(f"📤 Sending response to Node-RED:")
        print(f"   - recognized: {response['recognized']}")
        print(f"   - user_name: {response['user_name']}")
        print(f"   - confidence: {response['confidence']}%")
        print("="*50)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "recognized": False,
            "error": f"Server error: {str(e)}"
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Alternative endpoint that redirects to process-face"""
    return process_face()

@app.route('/test', methods=['GET'])
def test():
    """Health check endpoint"""
    return jsonify({
        "status": "working", 
        "message": "Face Recognition API is running!",
        "model_loaded": model is not None,
        "endpoints": {
            "POST /process-face": "Main face recognition endpoint",
            "POST /predict": "Alternative prediction endpoint", 
            "GET /test": "Health check"
        }
    })

@app.route('/', methods=['GET'])
def home():
    return """
    <html>
        <head>
            <title>Face Recognition API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-left: 4px solid #007cba; }
            </style>
        </head>
        <body>
            <h1>🎭 Face Recognition API</h1>
            <p>Status: <strong>🟢 Running</strong></p>
            <p>Model: <strong>{}</strong></p>
            
            <h2>Endpoints:</h2>
            <div class="endpoint">
                <strong>POST /process-face</strong> - Main face recognition endpoint (for Node-RED)<br>
                <em>Expects: {"image": "base64_string"}</em>
            </div>
            <div class="endpoint">
                <strong>POST /predict</strong> - Alternative prediction endpoint<br>
                <em>Same as /process-face</em>
            </div>
            <div class="endpoint">
                <strong>GET /test</strong> - Health check endpoint
            </div>
            <div class="endpoint">
                <strong>GET /</strong> - This page
            </div>
        </body>
    </html>
    """.format("✅ Loaded" if model else "❌ Not loaded")

# Initialize the application
def initialize_app():
    """Initialize the Flask application"""
    print("🚀 Initializing Face Recognition API...")
    print(f"🔮 TensorFlow version: {tf.__version__}")
    
    # Load model at startup
    load_model()
    
    print("📍 Available endpoints:")
    print("   POST http://localhost:5000/process-face")
    print("   POST http://localhost:5000/predict") 
    print("   GET  http://localhost:5000/test")
    print("   GET  http://localhost:5000/")
    print("🔧 Server is ready!")

# Initialize when the app starts
with app.app_context():
    initialize_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)