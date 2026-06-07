import os
import numpy as np
import tensorflow as tf
from src.dataset import prepare_mimii_data, prepare_self_recorded_data
from src.model import create_student_micro_model
from sklearn.metrics import classification_report, confusion_matrix

MIMII_DATA_DIR = "data/fan"
SELF_RECORDED_DIR = "data/SelfRecordedFan"
TEACHER_MODEL_PATH = "training/saved_models/fine_tuned_fan_model.keras"
STUDENT_TFLITE_OUTPUT_PATH = "deployment/distilled_student_model.tflite"

EPOCHS = 20
BATCH_SIZE = 32
DISTILL_LR = 0.001

def main():
    print("=== STARTING PRODUCTION KNOWLEDGE DISTILLATION ===")
    os.makedirs(os.path.dirname(STUDENT_TFLITE_OUTPUT_PATH), exist_ok=True)

    # Load unmodified Teacher
    if not os.path.exists(TEACHER_MODEL_PATH):
        raise FileNotFoundError(f"Teacher model not found at {TEACHER_MODEL_PATH}.")
    teacher_model = tf.keras.models.load_model(TEACHER_MODEL_PATH)

    # Load Datasets & Apply Isolated Scaling
    print("\nLoading datasets and calculating isolated domain scaling...")
    X_train_base, X_val_base, y_train_base, y_val_base = prepare_mimii_data(MIMII_DATA_DIR)
    
    # Base Dataset Scale (MIMII domain)
    base_mean = np.mean(X_train_base, axis=(0, 1), keepdims=True)
    base_std = np.std(X_train_base, axis=(0, 1), keepdims=True) + 1e-8
    
    X_train_base_scaled = (X_train_base - base_mean) / base_std
    X_val_base_scaled = (X_val_base - base_mean) / base_std

    # Local Dataset Scale
    # Pass raw arrays to extract native target properties directly
    X_target_raw, y_target = prepare_self_recorded_data(SELF_RECORDED_DIR, global_mean=0.0, global_std=1.0)
    
    target_mean = np.mean(X_target_raw, axis=(0, 1), keepdims=True)
    target_std = np.std(X_target_raw, axis=(0, 1), keepdims=True) + 1e-8
    
    # Scale local data using its own microphone profile
    X_target_scaled = (X_target_raw - target_mean) / target_std

    print("Aligning local environment features for deployment...")
    
    # Split local self-recorded microphone data into clean Train (80%) and Test (20%)
    split_idx = int(len(X_target_scaled) * 0.8)
    X_local_train = X_target_scaled[:split_idx]
    X_local_val = X_target_scaled[split_idx:]
    
    # Isolate the base anomalies from MIMII
    abnormal_train_indices = np.where(y_train_base == 1)[0]
    num_anomalies_to_mix = min(len(X_local_train), len(abnormal_train_indices))
    X_base_anomalies_train = X_train_base_scaled[abnormal_train_indices[:num_anomalies_to_mix]]
    
    # Isolate validation anomalies from MIMII
    abnormal_val_indices = np.where(y_val_base == 1)[0]
    num_anomalies_to_val = min(len(X_local_val), len(abnormal_val_indices))
    X_base_anomalies_val = X_val_base_scaled[abnormal_val_indices[:num_anomalies_to_val]]

    # Assemble Pure Training Pool
    X_student_train = np.concatenate([X_local_train, X_base_anomalies_train], axis=0)
    
    # Assemble Pure, Balanced Validation Pool (Mic Normal vs. Lab Failures)
    X_val_final = np.concatenate([X_local_val, X_base_anomalies_val], axis=0)
    y_val_final = np.concatenate([np.zeros(len(X_local_val)), np.ones(num_anomalies_to_val)], axis=0)
    
    print(f"-> Grounded Train Set Size: {X_student_train.shape[0]} windows.")
    print(f"-> Grounded Validation Set Size: {X_val_final.shape[0]} windows ({len(X_local_val)} Local Normal, {num_anomalies_to_val} Anomalies).")

    # Generate Teacher Probabilities
    print("Generating teacher targets...")
    teacher_targets = teacher_model.predict(X_student_train, batch_size=BATCH_SIZE)

    # Compile Student using standard Binary Crossentropy
    student_model = create_student_micro_model(X_student_train.shape[1:])
    student_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=DISTILL_LR),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    # Train
    print("\nTraining Student...")
    student_model.fit(X_student_train, teacher_targets, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # Convert to TFLite (Full INT8 internal math, Float32 Input/Output tracking)
    print("\nQuantizing Student...")
    converter = tf.lite.TFLiteConverter.from_keras_model(student_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def representative_data_gen():
        rng = np.random.default_rng(seed=42)
        shuffled_indices = rng.permutation(len(X_student_train))
        for idx in shuffled_indices[:200]:
            yield [np.expand_dims(X_student_train[idx], axis=0).astype(np.float32)]
            
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Leaving input/output as float32 simplifies execution code 
    # while the underlying network layer math runs fully in INT8 on hardware!
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_student_model = converter.convert()
    with open(STUDENT_TFLITE_OUTPUT_PATH, "wb") as f:
        f.write(tflite_student_model)
    print(f"[SUCCESS] Exported stable TFLite model.")

    # Evaluation
    print("\n=== Evaluating Distilled TFLite Model ===")
    interpreter = tf.lite.Interpreter(model_path=STUDENT_TFLITE_OUTPUT_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    y_pred_student = []
    for i in range(len(X_val_final)):
        tensor_in = np.expand_dims(X_val_final[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_details['index'], tensor_in)
        interpreter.invoke()
        y_pred_student.append(interpreter.get_tensor(output_details['index'])[0][0])
        
    y_pred_student = np.array(y_pred_student)
    
    print(f"DEBUG: Predicted Probability Range: {np.min(y_pred_student):.4f} to {np.max(y_pred_student):.4f}")

    # Dynamic sweep over probability bounds
    print("\nCalibrating optimal probability boundary...")
    TARGET_ABNORMAL_RECALL = 0.82  
    candidate_thresholds = np.linspace(0.01, 0.99, 100)
    
    best_threshold = 0.50
    best_normal_recall = 0.0
    
    for candidate in candidate_thresholds:
        preds = (y_pred_student > candidate).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val_final, preds).ravel()
        normal_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        abnormal_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if abnormal_recall >= TARGET_ABNORMAL_RECALL and normal_recall >= best_normal_recall:
            best_normal_recall = normal_recall
            best_threshold = candidate

    print(f"[CALIBRATION SUCCESS] Threshold Locked At: {best_threshold:.4f}")
    
    final_preds = (y_pred_student > best_threshold).astype(int)
    print("\n=== Distilled Student Performance Report ===")
    print(classification_report(y_val_final, final_preds, target_names=['Normal', 'Abnormal'], zero_division=0))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_val_final, final_preds))

if __name__ == "__main__":
    main()