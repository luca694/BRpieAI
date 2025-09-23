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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
for i in range(num_classes):
    train_count = np.sum(y_train == i)
    test_count = np.sum(y_test == i)
    print(f"Class {i}: {train_count} train samples, {test_count} test samples")

# Convert labels to categorical
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print(f"\n=== MODEL CONFIGURATION ===")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Number of classes: {num_classes}")

# Build model
model = Sequential([
    Conv2D(8, (3, 3), activation="relu", input_shape=(100, 100, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    Conv2D(16, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    Conv2D(32, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    
    Flatten(),
    Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

# Compile with lower learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=0.00001,
    verbose=1
)

# Training
batch_size = 2
epochs = 100

print("\n=== TRAINING STARTED ===")
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=batch_size),
    steps_per_epoch=max(1, len(X_train) // batch_size),
    epochs=epochs,
    validation_data=(X_test, y_test_cat),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n=== EVALUATION ===")
train_loss, train_accuracy = model.evaluate(X_train, y_train_cat, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Training Loss: {train_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Overfitting check
if train_accuracy > 0.95 and test_accuracy < 0.7:
    print("⚠️  WARNING: Model is overfitting!")
elif test_accuracy > 0.85:
    print("✅ Good performance!")
else:
    print("📊 Moderate performance")

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
print(confusion_matrix(y_true_classes, y_pred_classes))

# Save model if performance is reasonable
if test_accuracy > 0.6:
    model.save("model001.h5")
    print("✅ Model saved successfully!")
else:
    print("❌ Model performance too poor to save.")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

print("\n=== TRAINING COMPLETED ===")