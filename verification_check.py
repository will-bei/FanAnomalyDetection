# verification_check.py
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="deployment/final_fine_tuned_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
print(f"Verified Input Scale: {input_details['quantization'][0]}")