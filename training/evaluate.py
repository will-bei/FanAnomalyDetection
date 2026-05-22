# model evaluation file
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.dataset import prepare_mimii_data, prepare_self_recorded_data

MIMII_DATA_DIR = "data/fan"
SELF_RECORDED_DIR = "data/SelfRecordedFan"
FINE_TUNED_MODEL_PATH = "training/saved_models/fine_tuned_fan_model.keras"
OUTPUT_PLOT_PATH = "training/evaluation_results.png"

def evaluate_model():
    if not os.path.exists(FINE_TUNED_MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find model at '{FINE_TUNED_MODEL_PATH}'. Please run 'fine_tune.py' first."
        )
    model = tf.keras.models.load_model(FINE_TUNED_MODEL_PATH)
    
    # Load the MIMII validation split (contains both normal and abnormal data)
    _, X_val_mimii, _, y_val_mimii = prepare_mimii_data(MIMII_DATA_DIR)
    
    # Load your self-recorded dataset (contains only normal data)
    X_target, y_target = prepare_self_recorded_data(SELF_RECORDED_DIR)
    
    print(f"MIMII Validation Samples: {len(X_val_mimii)}")
    print(f"Self-Recorded Target Samples: {len(X_target)}")

    print("\nEvaluating on MIMII Benchmark Data...")
    # Generate raw probability predictions (0.0 to 1.0)
    y_pred_probs_mimii = model.predict(X_val_mimii)
    # Threshold at 0.5 to get binary decisions (0 = Normal, 1 = Abnormal)
    y_pred_mimii = (y_pred_probs_mimii > 0.5).astype(int).flatten()

    print("\nMIMII Classification Report:")
    print(classification_report(y_val_mimii, y_pred_mimii, target_names=["Normal", "Abnormal"]))
    
    cm_mimii = confusion_matrix(y_val_mimii, y_pred_mimii)

    print("\nEvaluating on Self-Recorded Target Fan...")
    if len(X_target) > 0:
        y_pred_probs_target = model.predict(X_target)
        y_pred_target = (y_pred_probs_target > 0.5).astype(int).flatten()
        
        # Calculate how often it incorrectly flags normal fan as an anomaly
        false_alarms = np.sum(y_pred_target == 1)
        false_alarm_rate = (false_alarms / len(X_target)) * 100
        
        print(f"Total Target Fan Windows Tested: {len(X_target)}")
        print(f"False Anomaly Alarms Triggered: {false_alarms}")
        print(f"Target Fan False Alarm Rate: {false_alarm_rate:.2f}%")
        
        # Merge datasets for a master consolidated Confusion Matrix
        X_combined = np.concatenate([X_val_mimii, X_target], axis=0)
        y_combined = np.concatenate([y_val_mimii, y_target], axis=0)
        y_pred_combined = np.concatenate([y_pred_mimii, y_pred_target], axis=0)
    else:
        print("[WARNING] No self-recorded data found to evaluate false alarm rates.")
        X_combined, y_combined, y_pred_combined = X_val_mimii, y_val_mimii, y_pred_mimii

    # Plot and Save Confusion Matrix
    print(f"\nPlotting Confusion Matrix to {OUTPUT_PLOT_PATH}...")
    cm_combined = confusion_matrix(y_combined, y_pred_combined)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_combined, display_labels=["Normal", "Abnormal"])
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    
    plt.title("Consolidated Performance Matrix\n(MIMII Split + Self-Recorded Fan)")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH)
    print("[SUCCESS] Evaluation complete.")

if __name__ == "__main__":
    evaluate_model()