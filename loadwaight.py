import tensorflow as tf
import numpy as np
import json
import pickle

# Load your model
model = tf.keras.models.load_model('model001.h5')

# 1. Extract all weights and biases
weights_dict = {}
for layer in model.layers:
    if layer.weights:  # Check if layer has weights
        layer_weights = layer.get_weights()
        weights_dict[layer.name] = {
            'kernel': layer_weights[0].tolist(),  # Convert to list for JSON
            'bias': layer_weights[1].tolist() if len(layer_weights) > 1 else None
        }

# 2. Extract model architecture/config
model_config = model.get_config()

# 3. Extract input/output shapes
input_shape = model.input_shape
output_shape = model.output_shape

# 4. Save everything to files
# Save weights as JSON
with open('model_weights.json', 'w') as f:
    json.dump(weights_dict, f)

# Save model configuration
with open('model_config.json', 'w') as f:
    json.dump(model_config, f)

# Save as pickle (preserves everything)
model_data = {
    'weights': weights_dict,
    'config': model_config,
    'input_shape': input_shape,
    'output_shape': output_shape
}

with open('complete_model_data.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✅ Everything extracted and saved!")
print(f"📊 Model input shape: {input_shape}")
print(f"📈 Model output shape: {output_shape}")
print(f"🏗️  Number of layers: {len(model.layers)}")