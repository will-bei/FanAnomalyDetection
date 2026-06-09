import os
import argparse
import numpy as np
import tensorflow as tf

from src.dataset_autoencoder import prepare_mimii_data, prepare_self_recorded_data

GAIN_CORRECTION = 2.419084 # CHANGE IN ACCORDANCE TO compute_gain_ratio.py

MIMII_DATA_ROOT = "data/fan"
NORMAL_DIR = "data/SelfRecordedFan/normal"
BASE_MODEL_PATH = "training/saved_models/mimii_autoencoder.keras"
FINE_TUNED_MODEL_PATH = "training/saved_models/fine_tuned_autoencoder.keras"
TFLITE_OUTPUT_PATH = "deployment/final_fine_tuned_model.tflite"

TARGET_H = 104
TARGET_W = 16

def enforce_shape(x):
    if x.shape[1:] != (104,16,1):
        raise ValueError(f"Bad shape: {x.shape}")
    return x

def main():
    parser = argparse.ArgumentParser(description="Fine-Tune TinyML Autoencoder Using id_00 Baseline Subset")
    parser.add_argument("--debug", action="store_true", help="Run short fine-tuning iteration for testing")
    args = parser.parse_args()

    epochs = 2 if args.debug else 15
    batch_size = 16

    # Constrain baseline loading to the id_00 folder to keep the process fast and lean
    baseline_sub_dir = os.path.join(MIMII_DATA_ROOT, "id_00")
    print(f"Loading Baseline 'id_00' Statistics for Data Calibration from: {baseline_sub_dir}")
    X_train_mimii, X_val_mimii, y_train_mimii, y_val_mimii, global_mean, global_std = prepare_mimii_data(baseline_sub_dir, gain_correction=GAIN_CORRECTION)
    
    if args.debug:
        X_train_mimii = X_train_mimii[:100]

    # Extract the exact mean and std format parameters from the baseline subset
    # global_mean = np.mean(X_train_mimii)
    # global_std  = np.std(X_train_mimii) + 1e-8

    X_self, _ = prepare_self_recorded_data(NORMAL_DIR)

    print(f"Loading and Preprocessing Target Self-Recorded Normal Data from: {NORMAL_DIR}")
    X_self = (X_self - global_mean) / global_std
    X_train_mimii = (X_train_mimii - global_mean) / global_std
    
    if len(X_self) == 0:
        raise FileNotFoundError(f"No self-recorded normal files could be parsed from {NORMAL_DIR}")

    X_self = enforce_shape(X_self.astype(np.float32))

    if args.debug:
        X_self = X_self[:20]

    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(f"Missing base model dependency: {BASE_MODEL_PATH}")

    print(f"\nLoading Base Autoencoder from {BASE_MODEL_PATH}...")

    base_model = tf.keras.models.load_model(BASE_MODEL_PATH)

    print("Re-compiling with a lowered learning rate for target domain fine-tuning...")
    base_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
        loss="mse",
        metrics=["mae"]
    )
    base_model.optimizer.learning_rate.assign(5e-5)
    print("\nFine-Tuning Architecture on Local Environmental Acoustics...")
    base_model.fit(
        X_self, X_self, # Train strictly on your local target audio
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1
    )

    os.makedirs(os.path.dirname(FINE_TUNED_MODEL_PATH), exist_ok=True)
    base_model.save(FINE_TUNED_MODEL_PATH)
    print(f"[SUCCESS] Fine-tuned model saved to {FINE_TUNED_MODEL_PATH}")

    print("\nConverting Fine-Tuned Model to Fully Quantized INT8 TFLite Blueprint...")
    def representative_dataset():
        # Determine sample size: use 50 or the number of items available, whichever is smaller
        n_mimii = min(len(X_train_mimii), 50)
        n_self = min(len(X_self), 50)
        
        idx_m = np.random.choice(len(X_train_mimii), n_mimii, replace=False)
        idx_s = np.random.choice(len(X_self), n_self, replace=False)

        pool = np.concatenate([
            X_train_mimii[idx_m],
            X_self[idx_s]
        ], axis=0)

        for i in range(len(pool)):
            yield [pool[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Force input/output to int8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    os.makedirs(os.path.dirname(TFLITE_OUTPUT_PATH), exist_ok=True)
    with open(TFLITE_OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"[SUCCESS] Final Fine-Tuned Quantized TFLite file exported to {TFLITE_OUTPUT_PATH}")

if __name__ == "__main__":
    main()