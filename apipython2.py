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
        print("Received data type:", type(data))
        print("Data keys:", list(data.keys()) if data else "No data")
        
        input_data = data['input']  # Base64 string
        print("Input data type:", type(input_data))
        print("Input data length:", len(input_data) if input_data else 0)
        print("Input preview:", input_data[:100] if input_data else "No data")
        
        # Remove data URL prefix if present
        if ',' in input_data:
            input_data = input_data.split(',')[1]
            print("After removing prefix, length:", len(input_data))
        
        # Decode base64 and convert to image
        print("Decoding base64...")
        image_data = base64.b64decode(input_data)
        print("Image data length:", len(image_data))
        
        print("Opening image...")
        image = Image.open(io.BytesIO(image_data))
        print("Image size:", image.size)
        print("Image mode:", image.mode)
        
        # Convert to grayscale and resize
        print("Converting to grayscale...")
        image = image.convert('L')
        print("Resizing to 100x100...")
        image = image.resize((100, 100))
        print("Final image size:", image.size)
        
        # Convert to numpy array and normalize
        print("Converting to numpy array...")
        input_array = np.array(image, dtype=np.float32) / 255.0
        print("Array shape:", input_array.shape)
        print("Array range:", input_array.min(), "to", input_array.max())
        
        input_array = input_array.reshape(1, 100, 100, 1)
        print("Reshaped array shape:", input_array.shape)
        
        print("Making prediction...")
        prediction = model.predict(input_array)
        print("Raw prediction:", prediction)
        
        # You'll need to adjust this based on your model output
        # For now, let's assume it's binary classification
        confidence = float(prediction[0][0])
        recognized = confidence > 0.5  # Adjust threshold as needed
        
        print("Confidence:", confidence)
        print("Recognized:", recognized)
        
        return jsonify({
            "success": True,
            "recognized": recognized,
            "confidence": confidence,
            "user_id": "user_1" if recognized else "unknown",
            "prediction": prediction.tolist()
        })
        
    except Exception as e:
        print("ERROR:", str(e))
        import traceback
        print("TRACEBACK:")
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