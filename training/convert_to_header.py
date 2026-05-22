import os

TFLITE_INPUT_PATH = "deployment/final_fine_tuned_model.tflite"
HEADER_OUTPUT_PATH = "deployment/model_data.h"

def bin_to_c_array(tflite_path, header_path):
    """
    Reads a TFLite binary file and exports it as a C++ char array header file
    compatible with TensorFlow Lite for Microcontrollers (TFLM).
    """
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(
            f"Could not find the TFLite file at '{tflite_path}'. "
            "Please run 'fine_tune.py' successfully before running this conversion script."
        )

    # Read the binary data from the TFLite model
    with open(tflite_path, "rb") as f:
        tflite_content = f.read()
    
    model_len = len(tflite_content)
    
    # Start constructing the header file string
    header_lines = [
        "#ifndef MODEL_DATA_H_",
        "#define MODEL_DATA_H_",
        "",
        "alignas(16) const unsigned char g_model_data[] = {",
    ]
    
    # ormat binary bytes into clean hexadecimal strings (12 bytes per row for readability)
    hex_bytes = []
    for i, byte in enumerate(tflite_content):
        hex_bytes.append(f"0x{byte:02x}")
        
        # Every 12 bytes, bundle them into a line with an indent
        if (i + 1) % 12 == 0:
            header_lines.append("    " + ", ".join(hex_bytes) + ",")
            hex_bytes = []
            
    # Catch any remaining trailing bytes
    if hex_bytes:
        header_lines.append("    " + ", ".join(hex_bytes))
    else:
        # Strip trailing comma from the last structured line if it was perfectly divisible
        header_lines[-1] = header_lines[-1].rstrip(",")
        
    # Add the footer and the explicit array length variable
    header_lines.extend([
        "};",
        "",
        f"const unsigned int g_model_data_len = {model_len};",
        "",
        "#endif // MODEL_DATA_H_"
    ])
    
    # Write the completed file
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    with open(header_path, "w") as f:
        f.write("\n".join(header_lines))
        
    print(f"[SUCCESS] C++ Model Header exported to: {header_path}")
    print(f"Model Byte Length: {model_len} bytes (~{model_len / 1024:.2f} KB)")

if __name__ == "__main__":
    bin_to_c_array(TFLITE_INPUT_PATH, HEADER_OUTPUT_PATH)