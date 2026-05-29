import numpy as np
from training.src.dataset import prepare_mimii_data

def inspect_features():
    print("=== PIPELINE SANITY CHECK ===")
    try:
        X_train, X_val, y_train, y_val = prepare_mimii_data("data/fan")
    except Exception as e:
        print(f"FAILED TO LOAD DATA: {e}")
        return

    print(f"\nArray Shapes:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # --- CHECK 1: ARE THE FEATURES ALL ZEROS OR EMTPY? ---
    print("\n--- Check 1: Feature Contents ---")
    print(f"  Max value in features: {np.max(X_train):.4f}")
    print(f"  Min value in features: {np.min(X_train):.4f}")
    print(f"  Mean value in features: {np.mean(X_train):.4f}")
    print(f"  Any NaNs? {np.isnan(X_train).any()}")

    # --- CHECK 2: ARE LABELS ACCURATELY DISTRIBUTED? ---
    print("\n--- Check 2: Label Verification ---")
    train_normal = np.sum(y_train == 0)
    train_abnormal = np.sum(y_train == 1)
    print(f"  Normal samples in Train: {train_normal}")
    print(f"  Abnormal samples in Train: {train_abnormal}")
    
    if train_normal == 0 or train_abnormal == 0:
        print("  [CRITICAL ERROR] One of your classes has 0 samples in the training split!")

    # --- CHECK 3: DO THE CLASSES LOOK IDENTICAL? ---
    print("\n--- Check 3: Feature Divergence ---")
    X_normal_mean = np.mean(X_train[y_train == 0]) if train_normal > 0 else 0
    X_abnormal_mean = np.mean(X_train[y_train == 1]) if train_abnormal > 0 else 0
    print(f"  Mean of Normal Features:   {X_normal_mean:.4f}")
    print(f"  Mean of Abnormal Features: {X_abnormal_mean:.4f}")
    
    diff = abs(X_normal_mean - X_abnormal_mean)
    print(f"  Absolute difference:       {diff:.6f}")
    if diff < 1e-4:
        print("  [WARNING] Normal and Abnormal arrays are mathematically identical. Your loop might be mixing up paths!")

if __name__ == "__main__":
    inspect_features()