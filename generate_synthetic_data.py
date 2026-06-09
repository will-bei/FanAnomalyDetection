import os
import glob
import numpy as np
from scipy.io import wavfile

# Define paths
NORMAL_DIR = os.path.join(".", "data", "SelfRecordedFan", "normal")
ABNORMAL_DIR = os.path.join(".", "data", "SelfRecordedFan", "abnormal")

# Create output directory if it doesn't exist
os.makedirs(ABNORMAL_DIR, exist_ok=True)

def load_wav(path):
    sample_rate, data = wavfile.read(path)
    # Convert to float32 normalized between -1.0 and 1.0 (matching standard processing)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    return sample_rate, data

def save_wav(path, sample_rate, data):
    # Convert back to Int16 for standard audio file writing
    clipped_data = np.clip(data, -1.0, 1.0)
    int16_data = (clipped_data * 32767.0).astype(np.int16)
    wavfile.write(path, sample_rate, int16_data)

# Anomaly Injection Functions
### noise functions generated using Gemini

def inject_clicks_and_rattles(data, sample_rate):
    """Simulates a loose blade or bearing hitting something periodically."""
    modified = data.copy()

    num_clicks = np.random.randint(5, 13)
    click_positions = np.linspace(0, len(modified) - 1, num_clicks, dtype=int)
    
    for pos in click_positions:
        pulse_len = int(sample_rate * 0.015)  # 15ms click duration
        if pos + pulse_len < len(modified):
            t = np.arange(pulse_len)
            pulse = np.sin(2 * np.pi * 3000 * t / sample_rate) * np.exp(-t / (pulse_len / 3))
            modified[pos:pos+pulse_len] += pulse * np.random.uniform(0.3, 0.6)
            
    return modified

def inject_bearing_friction(data):
    """Simulates grinding or constant bearing friction via continuous broadband noise."""
    modified = data.copy()
    # Generate white noise matching the length of the audio track
    noise = np.random.normal(0, 1, len(modified))
    # Apply a high-pass characteristic to mimic scratching/hissing metal
    noise = np.convolve(noise, [1, -0.9], mode='same') 
    # Normalize noise scale
    noise = noise / np.max(np.abs(noise))
    
    # Inject noise at a noticeable volume relative to the fan hum
    modified += noise * np.random.uniform(0.2, 0.4)
    return modified

def inject_motor_hum_shift(data, sample_rate):
    """Simulates electrical coil whine or mechanical imbalance by forcing an intense periodic resonance."""
    modified = data.copy()
    t = np.arange(len(modified)) / sample_rate
    # Generate a harsh, intrusive frequency tone (e.g., between 120Hz and 450Hz)
    fault_freq = np.random.uniform(120.0, 450.0)
    whine = np.sin(2 * np.pi * fault_freq * t)
    
    # Add subtle harmonic saturation to make it sound mechanical
    whine += 0.5 * np.sin(2 * np.pi * (fault_freq * 2) * t)
    
    modified += whine * np.random.uniform(0.2, 0.4)
    return modified

def main():
    abs_normal_dir = os.path.abspath(NORMAL_DIR)
    print(f"[DEBUG] Looking for files in absolute path: {abs_normal_dir}")
    
    search_path = os.path.join(abs_normal_dir, "*.wav")
    print(f"[DEBUG] Glob search query: {search_path}")
    
    normal_files = glob.glob(search_path)
    print(f"[DEBUG] Raw glob result array: {normal_files}")
    
    if not normal_files:
        print(f"Error: No .wav files found in {abs_normal_dir}")
        return

    print(f"Found {len(normal_files)} normal audio tracks. Generating anomalies...")
    
    generated_count = 0
    for file_path in normal_files:
        base_name = os.path.basename(file_path)
        print(f"[DEBUG] Processing file: {base_name}")
        name_without_ext = os.path.splitext(base_name)[0]
        
        try:
            sample_rate, raw_data = load_wav(file_path)
            print(f"  -> Loaded successfully. SR: {sample_rate}, Samples: {len(raw_data)}")
            
            click_variant = inject_clicks_and_rattles(raw_data, sample_rate)
            friction_variant = inject_bearing_friction(raw_data)
            hum_variant = inject_motor_hum_shift(raw_data, sample_rate)
            
            save_wav(os.path.join(ABNORMAL_DIR, f"{name_without_ext}_rattle.wav"), sample_rate, click_variant)
            save_wav(os.path.join(ABNORMAL_DIR, f"{name_without_ext}_friction.wav"), sample_rate, friction_variant)
            save_wav(os.path.join(ABNORMAL_DIR, f"{name_without_ext}_whine.wav"), sample_rate, hum_variant)
            
            generated_count += 3
            print(f"  -> Successfully generated 3 variants for {base_name}")
            
        except Exception as e:
            print(f"Skipping {base_name} due to processing error: {e}")

    print(f"Success! Generated {generated_count} synthetic anomaly profiles in {ABNORMAL_DIR}")

if __name__ == "__main__":
    main()