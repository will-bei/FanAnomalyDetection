import os
import argparse
import numpy as np
import tensorflow as tf

from src.dataset_autoencoder import prepare_mimii_data
from src.model import create_tiny_autoencoder

GAIN_CORRECTION = 2.419084 # CHANGE IN ACCORDANCE TO compute_gain_ratio.py

MIMII_DATA_ROOT = "data/fan"
MODEL_SAVE_DIR = "training/saved_models"
KERAS_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "mimii_autoencoder.keras")
TFLITE_OUTPUT_PATH = "deployment/model_data.tflite"

TARGET_H = 104
TARGET_W = 16

def enforce_shape(x):
    if len(x.shape) == 3:
        x = x[..., np.newaxis]

    assert x.shape[1:] == (104,16,1), f"Bad shape: {x.shape}"
    return x

def main():
    parser = argparse.ArgumentParser(description="Train Sub-Pixel Autoencoder Pipeline")
    parser.add_argument("--debug", action="store_true", help="Limit pipeline run to id_00 dataset subset")
    args = parser.parse_args()

    epochs = 2 if args.debug else 35
    batch_size = 32

    data_dir = os.path.join(MIMII_DATA_ROOT, "id_00") if args.debug else MIMII_DATA_ROOT
    print(f"Loading dataset located in: {data_dir}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(TFLITE_OUTPUT_PATH), exist_ok=True)

    X_train_all, X_val_all, y_train_all, y_val_all, global_mean, global_std = prepare_mimii_data(data_dir, gain_correction=GAIN_CORRECTION)

    if args.debug:
        X_train_all, X_val_all = X_train_all[:100], X_val_all[:20]
        y_train_all, y_val_all = y_train_all[:100], y_val_all[:20]

    # Filter for Normal operations first
    X_train = X_train_all[y_train_all == 0].astype(np.float32)
    X_val   = X_val_all[y_val_all == 0].astype(np.float32)

    # strict shape check
    def enforce_shape(x):
        assert x.shape[1:] == (104,16,1), f"Bad shape: {x.shape}"
        return x

    X_train = enforce_shape(X_train)
    X_val = enforce_shape(X_val)

    global_mean = np.mean(X_train, axis=0, keepdims=True)
    global_std  = np.std(X_train, axis=0, keepdims=True) + 1e-8

    X_train = (X_train - global_mean) / global_std
    X_val   = (X_val - global_mean) / global_std

    input_shape = X_train.shape[1:]
    
    print("\nInstantiating Sub-Pixel Autoencoder (Path A)...")
    model = create_tiny_autoencoder(input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    print("\nTraining Model Against High-Resolution Spectrograms...")
    model.fit(
        X_train, X_train, # Reconstruct the full high-resolution arrays directly
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    model.save(KERAS_MODEL_PATH)
    
    print("\nConverting base model to Fully Quantized INT8 TFLite...")
    def representative_dataset():
        for i in range(min(30, len(X_train))):
            yield [X_train[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # ensure every op is quantized to int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Force input/output to int8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(TFLITE_OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"[SUCCESS] Sub-Pixel Autoencoder TFLite exported: {TFLITE_OUTPUT_PATH}")

if __name__ == "__main__":
    main()