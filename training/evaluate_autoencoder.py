import os
import re
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from src.dataset_autoencoder import prepare_mimii_data, prepare_self_recorded_data

GAIN_CORRECTION = 2.419084 # CHANGE IN ACCORDANCE TO compute_gain_ratio.py

TARGET_H = 104
TARGET_W = 16

MIMII_DATA_ROOT = "data/fan"
NORMAL_DIR = "data/SelfRecordedFan/normal"
ABNORMAL_DIR = "data/SelfRecordedFan/abnormal"
TFLITE_MODEL_PATH = "deployment/final_fine_tuned_model.tflite"
OUTPUT_PLOT_PATH = "training/model_evaluation_results.png"
ARDUINO_SKETCH_PATH = os.path.join("deployment", "anomaly_detector_autoencoder", "anomaly_detector_autoencoder.ino")

def enforce_shape(x, target_h=TARGET_H, target_w=TARGET_W):
    h, w = x.shape[1], x.shape[2]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    x = np.pad(x, ((0,0),(0,pad_h),(0,pad_w),(0,0)), mode="constant")
    return x[:, :target_h, :target_w, :]

def update_arduino_threshold(sketch_path, new_threshold):
    if not os.path.exists(sketch_path):
        print(f"\n[WARNING] Arduino file skipped: {sketch_path}")
        return False
    try:
        with open(sketch_path, "r", encoding="utf-8") as f:
            content = f.read()
        pattern = r"(constexpr\s+float\s+kAnomalyThreshold\s*=\s*)([0-9.]+)(f?;)"
        if not re.search(pattern, content):
            return False
        content = re.sub(pattern, rf"\g<1>{new_threshold:.6f}f;", content)
        with open(sketch_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[SUCCESS] Arduino threshold locked: {new_threshold:.6f}f")
        return True
    except Exception as e:
        print(f"Failed updating parameter: {e}")
        return False

def calculate_reconstruction_errors(tflite_path, X_data):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale, in_zp = input_details["quantization"]
    out_scale, out_zp = output_details["quantization"]

    errors = []

    for i in range(len(X_data)):
        x = X_data[i:i+1].astype(np.float32)

        if in_scale != 0:
            x_in = np.round(x / in_scale + in_zp).astype(np.int8)
        else:
            x_in = x.astype(np.int8)

        interpreter.set_tensor(input_details["index"], x_in)
        interpreter.invoke()

        recon = interpreter.get_tensor(output_details["index"])

        if out_scale != 0:
            recon = (recon.astype(np.float32) - out_zp) * out_scale

        mse = np.mean((x - recon) ** 2)
        errors.append(mse)

    return np.array(errors)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Sub-Pixel Autoencoder Performance")
    parser.add_argument("--debug", action="store_true", help="Run short evaluations limited to id_00 subset")
    args = parser.parse_args()

    if not os.path.exists(TFLITE_MODEL_PATH):
        raise FileNotFoundError(f"Missing model: {TFLITE_MODEL_PATH}")

    mimii_dir = os.path.join(MIMII_DATA_ROOT, "id_00") if args.debug else MIMII_DATA_ROOT
    print(f"Evaluating dataset located in: {mimii_dir}")

    _, X_val, _, y_val, global_mean, global_std = prepare_mimii_data(mimii_dir, gain_correction=GAIN_CORRECTION)
    X_mimii_normal = X_val[y_val == 0]
    X_mimii_abnormal = X_val[y_val == 1]

    X_self_normal, _ = prepare_self_recorded_data(NORMAL_DIR, global_mean, global_std)
    X_self_abnormal, _ = prepare_self_recorded_data(ABNORMAL_DIR, global_mean, global_std)

    X_all_normal = np.concatenate([X_mimii_normal, X_self_normal], axis=0)
    X_all_abnormal = np.concatenate([X_mimii_abnormal, X_self_abnormal], axis=0)

    if args.debug:
        X_all_normal = X_all_normal[:20]
        X_all_abnormal = X_all_abnormal[:20]

    X_all_normal = enforce_shape(X_all_normal.astype(np.float32))
    X_all_abnormal = enforce_shape(X_all_abnormal.astype(np.float32))

    y_normal_true = np.zeros(len(X_all_normal), dtype=int)
    y_abnormal_true = np.ones(len(X_all_abnormal), dtype=int)
    y_combined = np.concatenate([y_normal_true, y_abnormal_true])

    print("\nRunning Quantized Inference passes...")
    mse_normal = calculate_reconstruction_errors(TFLITE_MODEL_PATH, X_all_normal)
    mse_abnormal = calculate_reconstruction_errors(TFLITE_MODEL_PATH, X_all_abnormal)
    
    y_probs_combined = np.concatenate([mse_normal, mse_abnormal])

    print("\nAutotuning threshold to aggressively prioritize Anomaly Catch Rate...")
    target_recall = 0.95
    best_percentile = 75
    best_threshold = np.percentile(mse_abnormal, 100 - (target_recall * 100))

    # for candidate_percentile in range(50, 87):
    #     candidate_threshold = np.percentile(mse_normal, candidate_percentile)
    #     preds = (y_probs_combined > candidate_threshold).astype(int)
        
    #     # Pull the full classification dictionary
    #     report = classification_report(y_combined, preds, output_dict=True, zero_division=0)
        
    #     # Extract individual metrics for the Abnormal class (label '1')
    #     abnormal_recall = report["1"]["recall"]
    #     abnormal_precision = report["1"]["precision"]
        
    #     # CUSTOM BIASED METRIC: Give Recall twice the weight of Precision to prioritize sensitivity
    #     if (abnormal_recall + abnormal_precision) > 0:
    #         weighted_score = (3 * abnormal_recall * abnormal_precision) / ( (2 * abnormal_precision) + abnormal_recall )
    #     else:
    #         weighted_score = 0.0
        
    #     if weighted_score > best_target_score:
    #         best_target_score = weighted_score
    #         best_percentile = candidate_percentile
    #         best_threshold = candidate_threshold

    # Load interpreter to inspect shapes
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print("\n--- TFLite Model Architecture Inspection ---")
    print(f"Input Name:    {input_details['name']}")
    print(f"Input Shape:   {input_details['shape']}")
    print(f"Input Type:    {input_details['dtype']}")
    print(f"Output Name:   {output_details['name']}")
    print(f"Output Shape:  {output_details['shape']}")
    print(f"Output Type:   {output_details['dtype']}")
    print("--------------------------------------------\n")

    # print(f"\n[BIASED HYPERPARAMETER LOCKED] Lowered Normal Percentile to: {best_percentile}%")
    print(f"[BIASED HYPERPARAMETER LOCKED] Resulting Lowered Threshold: {best_threshold:.6f}")

    # Apply our newly autotuned hyperparameter
    y_pred_final = (y_probs_combined > best_threshold).astype(int)
    
    # Add temporarily to evaluate_autoencoder.py main(), after mse_normal/mse_abnormal are computed:
    print(f"Normal   MSE — min: {mse_normal.min():.4f}  max: {mse_normal.max():.4f}  mean: {mse_normal.mean():.4f}")
    print(f"Abnormal MSE — min: {mse_abnormal.min():.4f}  max: {mse_abnormal.max():.4f}  mean: {mse_abnormal.mean():.4f}")
    # print(f"Selected threshold: {best_threshold:.6f} at percentile {best_percentile}")
    
    print("\nConsolidated Classification Report (Using Autotuned Sub-Pixel Thresholding):")
    print(classification_report(y_combined, y_pred_final, target_names=["Normal", "Abnormal"]))

    print(f"Plotting optimized Confusion Matrix to {OUTPUT_PLOT_PATH}...")
    cm_combined = confusion_matrix(y_combined, y_pred_final)
    
    os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_combined, display_labels=["Normal", "Abnormal"])
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    
    plt.title(f"Sub-Pixel Autoencoder Performance\n(Selected Threshold: {best_threshold:.6f})")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH)

    update_arduino_threshold(ARDUINO_SKETCH_PATH, best_threshold)
    # plt.show()

if __name__ == "__main__":
    main()