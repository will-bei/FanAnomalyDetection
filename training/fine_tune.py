import os
import numpy as np
import tensorflow as tf
from src.dataset import prepare_self_recorded_data
from src.model import compile_model

SELF_RECORDED_DIR = "data/SelfRecordedFan"    # Path to your self-recorded normal files
BASE_MODEL_PATH = "training/saved_models/mimii_base_model.keras"
FINE_TUNED_SAVE_DIR = "training/saved_models"
FINAL_TFLITE_OUTPUT_PATH = "deployment/final_fine_tuned_model.tflite"

EPOCHS = 10
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

    # Load and preprocess self-recorded fan data
    print("\nLoading Self-Recorded target fan data...")
    if not os.path.exists(SELF_RECORDED_DIR):
        raise FileNotFoundError(f"Could not find self-recorded data at '{SELF_RECORDED_DIR}'")
        
    X_target, y_target = prepare_self_recorded_data(SELF_RECORDED_DIR)
    print(f"Loaded {X_target.shape[0]} windows of target fan audio.")

    # 4. Fine-tune the model on the new baseline acoustics
    print("\nFine-Tuning the Network...")
    history = model.fit(
        X_target, y_target,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # 5. Save the fine-tuned Keras model
    ft_model_path = os.path.join(FINE_TUNED_SAVE_DIR, "fine_tuned_fan_model.keras")
    model.save(ft_model_path)
    print(f"\n[SUCCESS] Fine-tuned Keras model saved to: {ft_model_path}")

    # 6. Convert to final deployment-ready INT8 TFLite Model
    print("\nPost-Training Quantization to INT8...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset generator required for full INT8 quantization
    def representative_data_gen():
        # Provide a small sample of your data (e.g., 100 samples) to calibrate the weights
        num_calibration_steps = min(100, len(X_target))
        for i in range(num_calibration_steps):
            # Add batch dimension and ensure type matches float32
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