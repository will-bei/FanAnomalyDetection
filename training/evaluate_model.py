import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from training.src.dataset_model import prepare_mimii_data, prepare_self_recorded_data

MIMII_DATA_DIR = "data/fan"
SELF_RECORDED_DIR = "data/SelfRecordedFan"
FINE_TUNED_MODEL_PATH = "training/saved_models/fine_tuned_fan_model.keras"
OUTPUT_PLOT_PATH = "training/model_evaluation_results.png"

# Arduino auto-deployment: auto-update kAnomalyThreshold hyperparameter after evaluate.py complete
ARDUINO_SKETCH_PATH = os.path.join(".", "deployment", "anomaly_detector", "anomaly_detector.ino")

def update_arduino_threshold(sketch_path, new_threshold):
    """Searches the .ino file and rewrites the kAnomalyThreshold value."""
    if not os.path.exists(sketch_path):
        print(f"\n[WARNING] Could not find Arduino file at: {sketch_path}")
        print("Skipping automatic threshold deployment.")
        return False

    try:
        with open(sketch_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Regular expression looks for: constexpr float kAnomalyThreshold = X.XXf;
        # It accounts for varying spacing or missing 'f' suffix
        pattern = r"(constexpr\s+float\s+kAnomalyThreshold\s*=\s*)([0-9.]+)(f?;)"
        
        # Format the new value cleanly to two decimal places
        replacement = f"\\g<1>{new_threshold:.2f}f;"

        # Check if the variable actually exists in the file first
        if not re.search(pattern, content):
            print(f"\n[ERROR] Match failure: Could not find 'constexpr float kAnomalyThreshold' inside {sketch_path}")
            return False

        modified_content = re.sub(pattern, replacement, content)

        with open(sketch_path, "w", encoding="utf-8") as f:
            f.write(modified_content)

        print(f"\n[SUCCESS] Automatically patched Arduino firmware!")
        print(f"-> Location: {sketch_path}")
        print(f"-> Updated line to: constexpr float kAnomalyThreshold = {new_threshold:.2f}f;")
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to patch Arduino file: {e}")
        return False

def evaluate_model():
    if not os.path.exists(FINE_TUNED_MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find model at '{FINE_TUNED_MODEL_PATH}'. Please run 'fine_tune.py' first."
        )
    model = tf.keras.models.load_model(FINE_TUNED_MODEL_PATH)
    
    # Load validation and target splits
    _, X_val_mimii, _, y_val_mimii = prepare_mimii_data(MIMII_DATA_DIR)
    X_target, y_target = prepare_self_recorded_data(SELF_RECORDED_DIR)
    
    print(f"MIMII Validation Samples: {len(X_val_mimii)}")
    print(f"Self-Recorded Target Samples: {len(X_target)}")

    # Generate raw probability outputs
    print("\nGenerating raw model prediction probabilities...")
    y_pred_probs_mimii = model.predict(X_val_mimii).flatten()
    
    if len(X_target) > 0:
        y_pred_probs_target = model.predict(X_target).flatten()
        y_combined = np.concatenate([y_val_mimii, y_target], axis=0)
        y_probs_combined = np.concatenate([y_pred_probs_mimii, y_pred_probs_target], axis=0)
    else:
        y_combined = y_val_mimii
        y_probs_combined = y_pred_probs_mimii

    # Autotuning threshold
    print("\nInitializing Hyperparameter Optimization Engine...")
    
    # Sweeping from 0.01 to 0.99 in ultra-fine steps
    threshold_candidates = np.linspace(0.01, 0.99, 99)
    
    autotuned_threshold = 0.50
    best_target_metric = -1.0
    

    BETA_WEIGHT = 0.5 
    for t in threshold_candidates:
        preds = (y_probs_combined > t).astype(int)
        
        # Calculate scores for the 'Abnormal' class (index 1)
        precision, recall, f_beta, _ = precision_recall_fscore_support(
            y_combined, preds, beta=BETA_WEIGHT, average=None, labels=[0, 1], zero_division=0
        )
        
        abnormal_f_beta = f_beta[1]
        abnormal_precision = precision[1]
        normal_recall = recall[0] # Rate of correctly identified normal operations

        # Constraint Guardrail: Reject thresholds that completely kill our anomaly tracking
        if recall[1] < 0.60:
            continue

        # Optimize for the best F-beta balance
        if abnormal_f_beta > best_target_metric:
            best_target_metric = abnormal_f_beta
            autotuned_threshold = t

    print(f"\n[OPTIMIZER] Tuning Complete.")
    print(f"-> Mathematically Optimal Threshold: {autotuned_threshold:.2f}")
    print(f"-> Target Optimization Metric (F0.5-Score): {best_target_metric:.4f}")
    print("="*65)

    # Generate optimized final outputs using the autotuned hyperparameter
    y_pred_final = (y_probs_combined > autotuned_threshold).astype(int)
    
    print("\nConsolidated Classification Report (Using Autotuned Hyperparameter):")
    print(classification_report(y_combined, y_pred_final, target_names=["Normal", "Abnormal"]))

    # Save out the optimized Confusion Matrix
    print(f"Plotting optimized Confusion Matrix to {OUTPUT_PLOT_PATH}...")
    cm_combined = confusion_matrix(y_combined, y_pred_final)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_combined, display_labels=["Normal", "Abnormal"])
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    
    plt.title(f"Autotuned Performance Matrix\n(Selected Threshold: {autotuned_threshold:.2f})")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH)

    # Arduino Auto-patcher
    update_arduino_threshold(ARDUINO_SKETCH_PATH, autotuned_threshold)

    print("[SUCCESS] Evaluation complete.")

if __name__ == "__main__":
    evaluate_model()