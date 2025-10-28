from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model('model001.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("=== NEW PREDICT REQUEST ===")
        data = request.json
        
        # Debug: check if we have data
        if not data or 'input' not in data:
            return jsonify({
                "success": False,
                "error": "No input data received"
            }), 400
        
        input_data = data['input']  # Base64 string
        
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
        
        print("=== RAW MODEL OUTPUT ===")
        print("Prediction shape:", prediction.shape)
        print("Raw prediction values:", prediction)
        
        # Get the actual prediction
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        
        print("Predicted class:", predicted_class)
        print("Confidence:", confidence)
        
        # Map class indices to actual user names
        user_mapping = {
            0: "kb",      
            1: "kin",     
            2: "ryk"      
        }
        
        user_id = user_mapping.get(predicted_class, "unknown")
        recognized = confidence > 0.5
        
        print("Recognized user:", user_id)
        
        return jsonify({
            "success": True,
            "recognized": recognized,
            "confidence": confidence,
            "user_id": user_id,
            "predicted_class": int(predicted_class),
            "all_predictions": prediction.tolist()
        })
        
    except Exception as e:
        print("ERROR:", str(e))
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "working", "message": "Python API is running!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)