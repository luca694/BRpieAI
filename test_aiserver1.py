from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS
model = tf.keras.models.load_model('model001.h5')

# Serve the HTML page directly as a string (no file needed)
@app.route('/', methods=['GET'])
@app.route('/sign-in', methods=['GET'])
def serve_html():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign In</title>
    <style>
        body { background-color: rgb(255, 255, 255); margin: 0; padding: 0; display: flex; flex-direction: column; }
        .menu { position: fixed; top: 0; left: 0; width: 100%; padding: 14px 16px 10px; overflow: hidden;
                box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2); z-index: 1000; background-color: white; text-align: center; }
        .main-container { display: flex; margin-top: 80px; min-height: calc(100vh - 80px); width: 100%; }
        .camera-section { flex: 1; padding: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; }
        .info-section { flex: 1; padding: 20px; background-color: #f8f9fa; border-left: 2px solid #e9ecef; display: flex; flex-direction: column; }
        .cam video { width: 100%; max-width: 500px; height: 325px; border: 2px solid #c4c4c6; border-radius: 5px; 
                     object-fit: cover; transform: scaleX(-1); display: block; margin: 0 auto; }
        .btn { display: inline-block; padding: 7px 34px; margin: 4px; color: white; border-radius: 7px; 
               letter-spacing: 0.07cm; background-color: rgb(20, 24, 83); transition: all 0.3s ease; }
        .btn:hover { background-color: rgb(18, 17, 17); cursor: pointer; letter-spacing: 0.16cm; }
        .info-panel { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .status-indicator { padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin: 10px 0; }
        .status-waiting { background-color: #fff3cd; color: #856404; }
        .status-success { background-color: #d1ecf1; color: #0c5460; }
        .status-error { background-color: #f8d7da; color: #721c24; }
        .confidence-bar { height: 20px; background: #ecf0f1; border-radius: 10px; margin: 10px 0; overflow: hidden; }
        .confidence-fill { height: 100%; background: linear-gradient(90deg, #2ecc71, #27ae60); transition: width 0.5s ease; }
    </style>
</head>
<body>
    <div class="menu"><h1>FACE RECOGNITION - REGISTER</h1></div>
    <div class="main-container">
        <div class="camera-section">
            <div class="cam">
                <video id="video" width="360" height="360" autoplay></video>
                <br>
                <button onclick="captureForSignIn()" class="btn" style="margin-top: 20px;">SIGN IN WITH FACE</button>
            </div>
        </div>
        <div class="info-section">
            <div class="info-panel">
                <div class="info-title">User Information</div>
                <div id="userInfo">
                    <div class="status-indicator status-waiting" id="statusIndicator">Waiting for face recognition...</div>
                    <div class="info-item"><span class="info-label">User ID:</span> <span class="info-value" id="userId">-</span></div>
                    <div class="info-item"><span class="info-label">Name:</span> <span class="info-value" id="userName">-</span></div>
                    <div class="info-item"><span class="info-label">Role:</span> <span class="info-value" id="userRole">-</span></div>
                </div>
            </div>
            <div class="info-panel">
                <div class="info-title">Recognition Results</div>
                <div id="recognitionResults">
                    <div class="info-item"><span class="info-label">Confidence Level:</span> <span class="info-value" id="confidenceValue">0%</span></div>
                    <div class="confidence-bar"><div class="confidence-fill" id="confidenceBar" style="width: 0%"></div></div>
                    <div class="info-item"><span class="info-label">Status:</span> <span class="info-value" id="recognitionStatus">Not attempted</span></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const video = document.getElementById('video');
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => { video.srcObject = stream; })
                .catch(err => { console.error("Error accessing camera:", err); alert("Error accessing camera."); });
        }

        function updateStatus(message, type) {
            const indicator = document.getElementById('statusIndicator');
            indicator.textContent = message;
            indicator.className = `status-indicator status-${type}`;
        }

        async function captureForSignIn() {
            try {
                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                context.drawImage(video, 0, 0);
                const imageData = canvas.toDataURL('image/jpeg');
                
                updateStatus("Processing face...", "waiting");
                const button = document.querySelector('.btn');
                button.textContent = "PROCESSING...";
                button.disabled = true;
                
                // Send directly to Flask
                const response = await fetch('/process-face', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                });
                
                const result = await response.json();
                console.log("Result:", result);
                
                if (result.success && result.recognized) {
                    updateStatus(`Welcome ${result.user_name || 'User'}!`, "success");
                    document.getElementById('userId').textContent = result.user_id;
                    document.getElementById('userName').textContent = result.user_name || result.user_id;
                    document.getElementById('userRole').textContent = result.user_role || 'User';
                    document.getElementById('confidenceValue').textContent = result.confidence + '%';
                    document.getElementById('confidenceBar').style.width = result.confidence + '%';
                    document.getElementById('recognitionStatus').textContent = 'Recognized';
                } else {
                    updateStatus("Face not recognized", "error");
                    document.getElementById('recognitionStatus').textContent = 'Not recognized';
                }
                
            } catch (error) {
                console.error('Error:', error);
                updateStatus("Error processing face", "error");
            } finally {
                const button = document.querySelector('.btn');
                button.textContent = "SIGN IN WITH FACE";
                button.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

# Your face processing endpoint
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
            "confidence": round(confidence * 100, 2),
            "user_id": user_id,
            "user_name": user_id.upper(),
            "user_role": "User",
            "predicted_class": int(predicted_class)
        })
        
    except Exception as e:
        print("ERROR:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "working", "message": "Python API is running!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)