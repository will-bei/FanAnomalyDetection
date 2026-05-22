import os
import numpy as np
import tensorflow as tf
from src.dataset import prepare_mimii_data
from src.model import create_tiny_anomaly_model, compile_model

MIMII_DATA_DIR = "data/fan"             # MIMII dataset subset
MODEL_SAVE_DIR = "training/saved_models" 
TFLITE_OUTPUT_PATH = "deployment/model_data.tflite"

EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def main():
    # Check to ensure output directories exist
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(TFLITE_OUTPUT_PATH), exist_ok=True)

    # Load and preprocess the MIMII fan data
    print("Loading and Preprocessing MIMII Dataset...")
    if not os.path.exists(MIMII_DATA_DIR):
        raise FileNotFoundError(
            f"Could not find MIMII data directory at '{MIMII_DATA_DIR}'. "
            "Please ensure your data folder structure matches your plan."
        )
        
    X_train, X_val, y_train, y_val = prepare_mimii_data(MIMII_DATA_DIR)
    
    print(f"Training data shape:   {X_train.shape} (Labels: Normal={np.sum(y_train==0)}, Abnormal={np.sum(y_train==1)})")
    print(f"Validation data shape: {X_val.shape} (Labels: Normal={np.sum(y_val==0)}, Abnormal={np.sum(y_val==1)})")

    # Initialize and compile the model architecture
    print("\nInitializing TinyML CNN...")
    input_shape = X_train.shape[1:]  # (Time_Steps, MFCC_Features, 1)
    model = create_tiny_anomaly_model(input_shape)
    model = compile_model(model, learning_rate=LEARNING_RATE)
    model.summary()

    # Train the baseline model
    print("\nTraining Baseline Model...")
    
    # Early stopping prevents overfitting if validation loss stalls
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Save the desktop Keras model weights
    keras_model_path = os.path.join(MODEL_SAVE_DIR, "mimii_base_model.keras")
    model.save(keras_model_path)
    print(f"\n[SUCCESS] Baseline Keras model saved to: {keras_model_path}")

    # Convert to TFLite
    print("\nConverting Model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable default optimizations (this shrinks the float32 model weights)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    # Save the compressed TFLite binary file
    with open(TFLITE_OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)
        
    print(f"[SUCCESS] Compressed TFLite model exported to: {TFLITE_OUTPUT_PATH}")
    print(f"TFLite File Size: {os.path.getsize(TFLITE_OUTPUT_PATH) / 1024:.2f} KB")

if __name__ == "__main__":
    main()