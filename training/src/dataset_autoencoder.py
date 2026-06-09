import os
import glob
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

class AudioFeatureExtractor:
    def __init__(
        self,
        sample_rate=16000,
        duration=1.0,
        n_mels=16,          # IMPORTANT: match Arduino bands
        n_fft=1024,
        hop_length=1000,
        gain_correction=2.419084 # CHANGE IN ACCORDANCE TO compute_gain_ratio.py
    ):
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * duration)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.gain_correction = gain_correction

    def load_and_window_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        audio = audio * self.gain_correction
        audio = np.clip(audio, -1.0, 1.0)

        windows = []
        for i in range(0, len(audio) - self.window_samples + 1, self.window_samples):
            windows.append(audio[i:i + self.window_samples])

        return windows

    def extract_log_mel(self, audio_window):
        mel_spec = librosa.feature.melspectrogram(
            y=audio_window,
            sr=self.sample_rate,
            n_mels=104,   # MUST be fixed
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0
        )

        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # IMPORTANT: transpose to match (time, freq)
        log_mel = log_mel.T  # (time, n_mels)

        # Force exact shape (104,16)
        return log_mel[:, :self.n_mels]
    
    def extract_raw_mfcc_stack(self, audio_window):
        mfcc = librosa.feature.mfcc(
            y=audio_window,
            sr=self.sample_rate,
            n_mfcc=104,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        feature_map = log_mel[..., np.newaxis]

        # FORCE CORRECT ORIENTATION
        feature_map = np.transpose(feature_map, (0, 1, 2))  # no-op but clarifies intent

        # HARD ASSERT
        assert feature_map.shape[0] == 104, f"Freq mismatch: {feature_map.shape}"
        assert feature_map.shape[1] >= 16, f"Time mismatch: {feature_map.shape}"

        return feature_map[:, :16, :]
    
    def extract_raw_log_mel_stack(self, audio_window):
        mel = librosa.feature.melspectrogram(
            y=audio_window,
            sr=self.sample_rate,
            n_mels=104,
            n_fft=1024,
            hop_length=1000,
            power=2.0
        )

        log_mel = librosa.power_to_db(mel, ref=np.max)

        # HARD SHAPE GUARANTEE
        if log_mel.shape[0] != 104:
            log_mel = log_mel.T

        log_mel = log_mel[:, :16]

        return log_mel[..., np.newaxis]

def prepare_mimii_data(mimii_dir, test_size=0.2, random_state=42, gain_correction=1.0):
    extractor = AudioFeatureExtractor(gain_correction=gain_correction)

    X_list = []
    y_list = []

    clean_dir = mimii_dir.replace("\\", "/")
    search_pattern = f"{clean_dir}/**/*.wav"
    all_files = glob.glob(search_pattern, recursive=True)

    if not all_files:
        raise FileNotFoundError(f"No .wav files found in path: {search_pattern}")

    print(f"Found {len(all_files)} total audio files. Parsing explicit labels...")

    for path in all_files:
        normalized_path = path.replace("\\", "/").lower()

        # robust folder-based labeling (less brittle than substring matching)
        folder = os.path.basename(os.path.dirname(path)).lower()

        if "abnormal" in folder:
            label = 1
        elif "normal" in folder:
            label = 0
        else:
            continue

        try:
            windows = extractor.load_and_window_audio(path)
            # print(path, "windows:", len(windows))

            for window in windows:
                features = extractor.extract_raw_log_mel_stack(window)

                # ensure channel dimension exists (critical for CNN)
                if len(features.shape) == 2:
                    features = features[..., np.newaxis]

                X_list.append(features)
                y_list.append(label)

        except Exception as e:
            print(f"Skipping corrupt file {path}: {e}")

    if len(X_list) == 0:
        raise RuntimeError(
            "Dataset is empty after feature extraction. "
            "Check folder structure, labeling rules, or window size."
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"Total extracted samples: {len(X)}")
    print(f"Label distribution: {np.unique(y, return_counts=True)}")

    # stratified split (safe now because we ensured non-empty dataset)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("Computing Global Training Statistics...")

    global_mean = np.mean(X_train, axis=(0, 1, 2), keepdims=True)
    global_std = np.std(X_train, axis=(0, 1, 2), keepdims=True) + 1e-8

    X_train = (X_train - global_mean) / global_std
    X_val = (X_val - global_mean) / global_std

    return X_train, X_val, y_train, y_val, global_mean, global_std

def prepare_self_recorded_data(self_recorded_dir, global_mean=None, global_std=None):
    extractor = AudioFeatureExtractor()
    clean_dir = self_recorded_dir.replace("\\", "/")
    all_files = glob.glob(f"{clean_dir}/**/*.wav", recursive=True)
    
    X_list = []
    for path in all_files:
        try:
            windows = extractor.load_and_window_audio(path)
            for window in windows:
                features = extractor.extract_raw_log_mel_stack(window)
                X_list.append(features)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            
    X = np.array(X_list)
    
    if global_mean is not None and global_std is not None:
        X = (X - global_mean) / global_std
        
    y = np.zeros(len(X))
    return X, y