from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your model
try:
    model = tf.keras.models.load_model('model001.h5')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/process-face', methods=['POST'])
def process_face():
    try:
        print("🎯 Received face recognition request")
        
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data received"}), 400
        
        print(f"📦 Data keys: {list(data.keys())}")
        
        # Check for image data
        if 'image' not in data:
            return jsonify({"success": False, "error": "No image data in request"}), 400
        
        image_data = data['image']
        print(f"🖼️ Image data length: {len(image_data)}")
        
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            if ',' in image_data:
                image_data = image_data.split(',')[1]
                print("🔧 Removed data URL prefix")
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            print(f"📐 Original image size: {image.size}, mode: {image.mode}")
        except Exception as e:
            return jsonify({"success": False, "error": f"Image decoding failed: {str(e)}"}), 400
        
        # Preprocess image
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
                print("🔄 Converted to grayscale")
            
            # Resize to 100x100
            image = image.resize((100, 100))
            print(f"📏 Resized to: {image.size}")
            
            # Convert to numpy array
            img_array = np.array(image, dtype=np.float32) / 255.0
            print(f"🔢 Array shape: {img_array.shape}, range: [{np.min(img_array):.3f}, {np.max(img_array):.3f}]")
            
            # Reshape for model
            input_array = img_array.reshape(1, 100, 100, 1)
            print(f"🎯 Final input shape: {input_array.shape}")
            
        except Exception as e:
            return jsonify({"success": False, "error": f"Image processing failed: {str(e)}"}), 400
        
        # Make prediction
        if model is None:
            return jsonify({"success": False, "error": "Model not loaded"}), 500
        
        try:
            prediction = model.predict(input_array, verbose=0)
            print(f"🤖 Prediction shape: {prediction.shape}")
            print(f"🔍 Raw predictions: {prediction}")
            
            # Get results
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])
            
            print(f"🎭 Predicted class: {predicted_class}")
            print(f"📊 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            
        except Exception as e:
            return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500
        
        # User mapping
        user_mapping = {
            0: {"user_id": "kb", "user_name": "Khoo Ben", "user_st": "2291", "grade": "11"},
            1: {"user_id": "kin", "user_name": "Kin Seng", "user_st": "6767", "grade": "11"},
            2: {"user_id": "ryk", "user_name": "Ryan Koh", "user_st": "8686", "grade": "11"}
        }
        
        # Get user info
        user_info = user_mapping.get(predicted_class, {
            "user_id": "unknown",
            "user_name": "Unknown User", 
            "user_st": "N/A",
            "grade": "N/A"
        })
        
        # Determine recognition
        recognized = confidence > 0.5
        status = "✅ Recognized" if recognized else "❌ Not recognized"
        print(f"{status} - User: {user_info['user_name']} (Confidence: {confidence*100:.2f}%)")
        
        # Prepare response
        response = {
            "success": True,
            "recognized": recognized,
            "confidence": round(confidence * 100, 2),
            "user_id": user_info["user_id"],
            "user_name": user_info["user_name"],
            "user_st": user_info["user_st"],
            "grade": user_info["grade"],
            "predicted_class": int(predicted_class),
            "message": f"Recognized: {user_info['user_name']}" if recognized else "Face not recognized"
        }
        
        print(f"📤 Sending response: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500

@app.route('/test', methods=['GET', 'POST'])
def test():
    return jsonify({
        "status": "working", 
        "message": "Python AI API is running!",
        "endpoints": {
            "POST /process-face": "Face recognition",
            "GET /test": "Health check"
        }
    })

@app.route('/', methods=['GET'])
def home():
    return """
    <html>
        <body>
            <h1>Face Recognition API</h1>
            <p>Endpoints:</p>
            <ul>
                <li>POST /process-face - Process face image</li>
                <li>GET /test - Health check</li>
            </ul>
        </body>
    </html>
    """

if __name__ == '__main__':
    print("🚀 Starting Face Recognition API...")
    print("📍 Endpoints:")
    print("   POST http://localhost:5000/process-face")
    print("   GET  http://localhost:5000/test")
    print("   GET  http://localhost:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True)