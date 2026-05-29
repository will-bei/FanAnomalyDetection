import os
import glob
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, duration=1.0, n_mfcc=13, n_fft=400, hop_length=160):
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * duration)
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def load_and_window_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        windows = []
        for i in range(0, len(audio) - self.window_samples + 1, self.window_samples):
            windows.append(audio[i : i + self.window_samples])
        return windows

    def extract_raw_mfcc_stack(self, audio_window):
        mfcc = librosa.feature.mfcc(
            y=audio_window, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mfcc, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        feature_map = np.stack([mfcc, delta, delta2], axis=-1)
        return np.transpose(feature_map, (1, 0, 2))

def prepare_mimii_data(mimii_dir, test_size=0.2, random_state=42):
    extractor = AudioFeatureExtractor()
    
    X_list = []
    y_list = []
    
    # 1. Recursively find ALL .wav files inside the directory tree
    # Replacing backslashes ensures Windows/Mac uniformity for glob parsing
    clean_dir = mimii_dir.replace("\\", "/")
    search_pattern = f"{clean_dir}/**/*.wav"
    all_files = glob.glob(search_pattern, recursive=True)
    
    if not all_files:
        raise FileNotFoundError(f"No .wav files found in path: {search_pattern}")
        
    print(f"Found {len(all_files)} total audio files. Parsing explicit labels...")

    # 2. Inspect every single file path explicitly to determine class
    for path in all_files:
        normalized_path = path.replace("\\", "/").lower()
        
        # Explicit string checks prevent path-bleed bugs
        if "/normal/" in normalized_path:
            label = 0
        elif "/abnormal/" in normalized_path:
            label = 1
        else:
            # Skip files that don't belong to a core split
            continue
            
        try:
            windows = extractor.load_and_window_audio(path)
            for window in windows:
                features = extractor.extract_raw_mfcc_stack(window)
                X_list.append(features)
                y_list.append(label)
        except Exception as e:
            print(f"Skipping corrupt file {path}: {e}")

    X = np.array(X_list)
    y = np.array(y_list)

    # 3. Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 4. Apply Global Training Scale Calibration
    print("Computing Explicit Global Training Statistics...")
    global_mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    global_std = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-8
    
    X_train = (X_train - global_mean) / global_std
    X_val = (X_val - global_mean) / global_std
    
    return X_train, X_val, y_train, y_val

def prepare_self_recorded_data(self_recorded_dir, global_mean=None, global_std=None):
    extractor = AudioFeatureExtractor()
    clean_dir = self_recorded_dir.replace("\\", "/")
    all_files = glob.glob(f"{clean_dir}/**/*.wav", recursive=True)
    
    X_list = []
    for path in all_files:
        try:
            windows = extractor.load_and_window_audio(path)
            for window in windows:
                features = extractor.extract_raw_mfcc_stack(window)
                X_list.append(features)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            
    X = np.array(X_list)
    
    if global_mean is not None and global_std is not None:
        X = (X - global_mean) / global_std
        
    y = np.zeros(len(X))
    return X, y