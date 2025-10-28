from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for Node-RED
model = tf.keras.models.load_model('model001.h5')

# AI API endpoint only - no HTML
@app.route('/process-face', methods=['POST'])
def process_face():
    try:
        print("=== NEW PREDICT REQUEST ===")
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "No image data received"
            }), 400
        
        input_data = data['image']
        
        # Remove data URL prefix if present
        if ',' in input_data:
            input_data = input_data.split(',')[1]
        
        # Decode base64 and convert to image
        image_data = base64.b64decode(input_data)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale and resize
        image = image.convert('L')
        image = image.resize((100, 100))
        
        # Convert to numpy array and normalize
        input_array = np.array(image, dtype=np.float32) / 255.0
        input_array = input_array.reshape(1, 100, 100, 1)
        
        print("Input array shape:", input_array.shape)
        
        # Make prediction
        prediction = model.predict(input_array)
        
        print("Raw prediction values:", prediction)
        
        # Get the actual prediction
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        
        print("Predicted class:", predicted_class)
        print("Confidence:", confidence)
        
        # Map class indices to actual user information
        user_mapping = {
            0: {
                "user_id": "kb",
                "user_name": "Khoo Ben", 
                "user_st": "2291",
                "grade": "11",
                
            },
            1: {
                "user_id": "kin", 
                "user_name": "Kin Seng",
                "user_st": "6767",
                "grade": "11",
                
            },
            2: {
                "user_id": "ryk",
                "user_name": "Ryan Koh", 
                "user_st": "8686",
                "grade": "11",
                
            }
        }
        
        user_info = user_mapping.get(predicted_class, {
            "user_id": "unknown",
            "user_name": "Unknown User",
            "user_st": "Unknown", 
            "grade": "Unknown",
        })
        
        recognized = confidence > 0.5
        
        print("Recognized user:", user_info["user_name"])
        
        return jsonify({
            "success": True,
            "recognized": recognized,
            "confidence": round(confidence * 100, 2),
            "user_id": user_info["user_id"],
            "user_name": user_info["user_name"],
            "user_st": user_info["user_st"],
            "grade": user_info["grade"],
            "predicted_class": int(predicted_class)
        })
        
    except Exception as e:
        print("ERROR:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route('/predict', methods=['POST'])
def predict():
    # Call the same function as process-face
    return process_face()

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "working", "message": "Python AI API is running!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)