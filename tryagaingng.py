from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
import os
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your model
try:
    model = tf.keras.models.load_model('model001.h5')
    print("✅ Model loaded successfully")
    # Print model info to debug
    print(f"📊 Input shape: {model.input_shape}")
    print(f"📈 Output shape: {model.output_shape}")
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
        
        # Check for image data
        if 'image' not in data:
            return jsonify({"success": False, "error": "No image data in request"}), 400
        
        image_data = data['image']
        
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            if ',' in image_data:
                image_data = image_data.split(',')[1]
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({"success": False, "error": f"Image decoding failed: {str(e)}"}), 400
        
        # Preprocess image
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize to 100x100
            image = image.resize((100, 100))
            
            # Convert to numpy array
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            # Reshape for model - FIXED: Ensure correct shape
            input_array = img_array.reshape(1, 100, 100, 1)
            print(f"🎯 Final input shape: {input_array.shape}")
            
        except Exception as e:
            return jsonify({"success": False, "error": f"Image processing failed: {str(e)}"}), 400
        
        # Make prediction - FIXED: Use correct arguments
        if model is None:
            return jsonify({"success": False, "error": "Model not loaded"}), 500
        
        try:
            # ✅ CORRECT: Only use valid arguments
            prediction = model.predict(input_array, verbose=0)
            # OR with batch_size if needed:
            # prediction = model.predict(input_array, batch_size=1, verbose=0)
            
            print(f"🤖 Prediction shape: {prediction.shape}")
            
            # Get results
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])
            
            print(f"🎭 Predicted class: {predicted_class}")
            print(f"📊 Confidence: {confidence:.4f}")
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500
        
        # User mapping and response (your existing code)
        user_mapping = {
            0: {"user_id": "kb", "user_name": "Khoo Ben", "user_st": "2291", "grade": "11"},
            1: {"user_id": "kin", "user_name": "Kin Seng", "user_st": "6767", "grade": "11"},
            2: {"user_id": "ryk", "user_name": "Ryan Koh", "user_st": "8686", "grade": "11"}
        }
        
        user_info = user_mapping.get(predicted_class, {
            "user_id": "unknown", "user_name": "Unknown User", "user_st": "N/A", "grade": "N/A"
        })
        
        recognized = confidence > 0.5
        
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
        
        return jsonify(response)
        
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500

# Your other routes remain the same...