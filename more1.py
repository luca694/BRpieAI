import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import your data
from dat import X_train, X_test, y_train, y_test, y

print("=== DATA DEBUGGING ===")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train: {y_train}")
print(f"y_test: {y_test}")
print(f"Unique labels in y_train: {np.unique(y_train)}")
print(f"Unique labels in y_test: {np.unique(y_test)}")
print(f"X_train range: [{X_train.min()}, {X_train.max()}]")

# Add channel dimension if missing and normalize
if len(X_train.shape) == 3:
    X_train = X_train.reshape(X_train.shape[0], 100, 100, 1)
    X_test = X_test.reshape(X_test.shape[0], 100, 100, 1)
    print(f"Reshaped X_train: {X_train.shape}")

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
print("Data normalized to [0, 1]")

# Verify class distribution
num_classes = len(np.unique(y))
print(f"Number of classes: {num_classes}")

for i in range(num_classes):
    train_count = np.sum(y_train == i)
    test_count = np.sum(y_test == i)
    print(f"Class {i}: {train_count} train samples, {test_count} test samples")

# Check for class imbalance
class_counts = [np.sum(y_train == i) for i in range(num_classes)]
min_samples = min(class_counts)
max_samples = max(class_counts)
if max_samples / min_samples > 3:  # Significant imbalance
    print(f"⚠️ Class imbalance detected! Ratio: {max_samples/min_samples:.2f}")

# Convert labels to categorical
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print(f"\n=== IMPROVED MODEL CONFIGURATION ===")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Improved model architecture
model = Sequential([
    # First Conv Block
    Conv2D(32, (3, 3), activation="relu", input_shape=(100, 100, 1), padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation="relu", padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Second Conv Block
    Conv2D(64, (3, 3), activation="relu", padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation="relu", padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Third Conv Block
    Conv2D(128, (3, 3), activation="relu", padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Fourth Conv Block
    Conv2D(256, (3, 3), activation="relu", padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Dense Layers
    Flatten(),
    Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

# Compile with adaptive learning rate
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])
model.summary()

# Enhanced data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Enhanced callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=25,
    restore_best_weights=True,
    verbose=1,
    min_delta=0.001
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=0.000001,
    verbose=1
)

# Model checkpoint to save best model regardless of threshold
model_checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Adjust batch size based on dataset size
batch_size = min(32, len(X_train) // 10)  # Adaptive batch size
batch_size = max(2, batch_size)  # Minimum batch size of 2
epochs = 200

print(f"\nUsing batch size: {batch_size}")
print(f"Training for maximum {epochs} epochs")

print("\n=== TRAINING STARTED ===")
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=batch_size),
    steps_per_epoch=max(1, len(X_train) // batch_size),
    epochs=epochs,
    validation_data=(X_test, y_test_cat),
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# Load the best model
from keras.models import load_model
model = load_model('best_model.h5')
print("✅ Loaded best model from training")

print("\n=== EVALUATION ===")
train_loss, train_accuracy = model.evaluate(X_train, y_train_cat, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Training Loss: {train_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Better performance analysis
accuracy_gap = train_accuracy - test_accuracy
if accuracy_gap > 0.2:
    print("⚠️  WARNING: Model is overfitting!")
elif test_accuracy > 0.85:
    print("✅ Excellent performance!")
elif test_accuracy > 0.75:
    print("✅ Good performance!")
elif test_accuracy > 0.6:
    print("📊 Moderate performance - needs improvement")
else:
    print("❌ Poor performance - consider data/architecture changes")

# Detailed prediction analysis
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_cat, axis=1)

print("\n=== PREDICTION ANALYSIS ===")
print("True labels:", y_true_classes)
print("Pred labels:", y_pred_classes)

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes))

print("Confusion Matrix:")
cm = confusion_matrix(y_true_classes, y_pred_classes)
print(cm)

# Calculate per-class accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)
print("\nPer-class accuracy:")
for i, acc in enumerate(class_accuracy):
    print(f"Class {i}: {acc:.3f}")

# Save model with performance-based naming
if test_accuracy > 0.6:  # Lowered threshold
    model_name = f"model_acc_{test_accuracy:.3f}.h5"
    model.save(model_name)
    print(f"✅ Model saved as {model_name}")
else:
    print("❌ Model performance too poor to save.")
    # But we already saved the best one during training

# Enhanced plotting
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
# Plot learning rate
if 'lr' in history.history:
    plt.plot(history.history['lr'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.ylabel('LR')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
else:
    # Plot accuracy difference
    acc_diff = [train - val for train, val in zip(history.history['accuracy'], history.history['val_accuracy'])]
    plt.plot(acc_diff, label='Train-Val Accuracy Gap', color='red')
    plt.title('Overfitting Indicator')
    plt.ylabel('Accuracy Gap')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('improved_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional diagnostics
print("\n=== ADDITIONAL DIAGNOSTICS ===")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Training stopped at epoch: {len(history.history['accuracy'])}")

# Check if we need more data
if len(X_train) < 1000 and test_accuracy < 0.7:
    print("💡 Suggestion: Consider collecting more training data")
if num_classes > 5 and len(X_train) < 500 * num_classes:
    print("💡 Suggestion: You might need more data for the number of classes")