import os
import numpy as np
import tensorflow as tf
from training.src.dataset_model import prepare_self_recorded_data
from src.model import compile_model

# Updated Path Structure that uses synthetic data
NORMAL_DIR = "data/SelfRecordedFan/normal"
ABNORMAL_DIR = "data/SelfRecordedFan/abnormal"

BASE_MODEL_PATH = "training/saved_models/mimii_base_model.keras"
FINE_TUNED_SAVE_DIR = "training/saved_models"
FINAL_TFLITE_OUTPUT_PATH = "deployment/final_fine_tuned_model.tflite"

# Increased epochs slightly as binary classification across multiple fault modes
# benefits from a few extra parameter tuning iterations.
EPOCHS = 15
BATCH_SIZE = 16
FINE_TUNE_LR = 0.0001

def main():
    os.makedirs(FINE_TUNED_SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FINAL_TFLITE_OUTPUT_PATH), exist_ok=True)

    # Load the pre-trained Keras baseline model
    print("Loading Pre-trained MIMII Baseline Model...")
    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(
            f"Baseline model not found at '{BASE_MODEL_PATH}'. "
            "Please run 'train.py' first to establish the base weights."
        )
    
    model = tf.keras.models.load_model(BASE_MODEL_PATH)
    print("Base model loaded successfully.")

    # Freeze the feature extraction layers
    print("\nFreezing Convolutional Core layers...")
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer.trainable = False
            print(f"Layer Frozen: {layer.name}")
        else:
            layer.trainable = True
            print(f"Layer Kept Trainable: {layer.name}")

    # Re-compile the model to apply the freezing and a tighter learning rate
    model = compile_model(model, learning_rate=FINE_TUNE_LR)

    # New Subfolder logic
    print("\nLoading Self-Recorded target fan data...")
    if not os.path.exists(NORMAL_DIR) or not os.path.exists(ABNORMAL_DIR):
        raise FileNotFoundError(
            f"Could not find expected data directories. Make sure both "
            f"'{NORMAL_DIR}' and '{ABNORMAL_DIR}' exist."
        )
        
    # Load Normal audio tracks (Label 0)
    print("Processing normal ambient profiles...")
    X_normal, _ = prepare_self_recorded_data(NORMAL_DIR)
    y_normal = np.zeros(X_normal.shape[0], dtype=np.float32)
    print(f"  -> Loaded {X_normal.shape[0]} normal feature windows.")

    # Load Synthetic Abnormal tracks (Label 1)
    print("Processing synthetic anomaly tracks...")
    X_abnormal, _ = prepare_self_recorded_data(ABNORMAL_DIR)
    y_abnormal = np.ones(X_abnormal.shape[0], dtype=np.float32)
    print(f"  -> Loaded {X_abnormal.shape[0]} abnormal feature windows.")

    # Merge arrays to create a unified dataset
    X_target = np.concatenate([X_normal, X_abnormal], axis=0)
    y_target = np.concatenate([y_normal, y_abnormal], axis=0)

    # Shuffle dataset to ensure optimal gradient descent batches
    indices = np.arange(X_target.shape[0])
    np.random.shuffle(indices)
    X_target = X_target[indices]
    y_target = y_target[indices]

    print(f"\nFinal training dataset dimensions: {X_target.shape}")
    print(f"Total Windows: {X_target.shape[0]} ({np.sum(y_target==0)} Normal, {np.sum(y_target==1)} Abnormal)")

    # Fine-tune the model on the new baseline acoustics
    print("\nFine-Tuning the Network...")
    history = model.fit(
        X_target, y_target,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Save the fine-tuned Keras model
    ft_model_path = os.path.join(FINE_TUNED_SAVE_DIR, "fine_tuned_fan_model.keras")
    model.save(ft_model_path)
    print(f"\n[SUCCESS] Fine-tuned Keras model saved to: {ft_model_path}")

    # Convert to final deployment-ready INT8 TFLite Model
    print("\nPost-Training Quantization to INT8...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset generator required for full INT8 quantization
    def representative_data_gen():
        num_calibration_steps = min(100, len(X_target))
        for i in range(num_calibration_steps):
            yield [X_target[i:i+1].astype(np.float32)]

    converter.representative_dataset = representative_data_gen
    
    # Force full integer quantization (crucial for Arduino HW acceleration)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    # Save the compressed INT8 TFLite model
    with open(FINAL_TFLITE_OUTPUT_PATH, "wb") as f:
        f.write(tflite_quant_model)
        
    print(f"[SUCCESS] Final Full INT8 TFLite model exported to: {FINAL_TFLITE_OUTPUT_PATH}")
    print(f"Quantized TFLite File Size: {os.path.getsize(FINAL_TFLITE_OUTPUT_PATH) / 1024:.2f} KB")

if __name__ == "__main__":
    main()