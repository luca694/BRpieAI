from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('model001.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = data['input']
    
    # Convert to numpy and ensure exact shape (1, 100, 100, 1)
    input_array = np.array(input_data, dtype=np.float32)
    
    # Reshape to exact model requirements
    input_array = input_array.reshape(1, 100, 100, 1)
    
    prediction = model.predict(input_array)
    
    return jsonify({
        "success": True,
        "prediction": prediction.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)