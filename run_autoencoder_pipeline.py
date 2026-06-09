import os
import sys
import subprocess

def run_script(script_path):
    """Executes a python script and streams its output in real-time."""
    print("=" * 70)
    print(f"RUNNING: {script_path}")
    print("=" * 70)
    
    # Normalize path for the current operating system
    normalized_path = os.path.normpath(script_path)
    
    if not os.path.exists(normalized_path):
        print(f"[ERROR] Script not found at target location: {normalized_path}")
        return False

    try:
        # Run the script and stream stdout/stderr directly to the terminal
        process = subprocess.Popen(
            [sys.executable, normalized_path],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        
        # Wait for the process to complete
        process.communicate()
        
        if process.returncode == 0:
            print(f"\n[SUCCESS] Completed: {script_path}\n")
            return True
        else:
            print(f"\n[FAILURE] Script exited with error code {process.returncode}: {script_path}\n")
            return False
            
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline execution failed for {script_path}: {e}\n")
        return False

def main():
    pipeline = [
        "generate_synthetic_data.py",
        "training/train_autoencoder.py",
        "training/fine_tune_autoencoder.py",
        "training/evaluate_autoencoder.py",
        "training/convert_to_header_autoencoder.py",
        "export_arduino_constants.py",
        "verification_check.py"
    ]
    
    print("Starting TinyML Acoustic Autoencoder Pipeline Automation...\n")
    
    for script in pipeline:
        success = run_script(script)
        if not success:
            print("=" * 70)
            print("PIPELINE HALTED: An error occurred in a previous step.")
            print("=" * 70)
            sys.exit(1)
            
    print("=" * 70)
    print("ALL STAGES COMPLETE: Model is trained, quantized, and deployment patched!")
    print("=" * 70)

if __name__ == "__main__":
    main()