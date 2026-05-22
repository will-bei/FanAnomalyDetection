import os
import glob
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, duration=1.0, n_mfcc=13, n_fft=400, hop_length=160):
        """
        Default parameters are optimized for TinyML (16kHz audio, 25ms frame, 10ms hop).
        This results in a predictable feature map size for the microcontroller.
        """
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * duration)
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def load_and_window_audio(self, file_path):
        """Loads an audio file and slices it into fixed-duration windows."""
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        
        # Split long audio into distinct 1-second chunks
        windows = []
        for i in range(0, len(audio) - self.window_samples + 1, self.window_samples):
            windows.append(audio[i : i + self.window_samples])
            
        return windows

    def extract_mfcc(self, audio_window):
        """Extracts log-mel MFCC features from a single audio window."""
        mfcc = librosa.feature.mfcc(
            y=audio_window, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mfcc, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        # Transpose to make time the major axis: (Time_Steps, Features)
        # This aligns better with 1D Convolutions or MobileNet-style layouts
        return mfcc.T

    def process_directory(self, base_path, pattern="**/*.wav"):
        """Recursively finds all wav files and extracts features."""
        search_path = os.path.join(base_path, pattern)
        file_paths = glob.glob(search_path, recursive=True)
        
        features = []
        for path in file_paths:
            try:
                windows = self.load_and_window_audio(path)
                for window in windows:
                    mfcc_feat = self.extract_mfcc(window)
                    features.append(mfcc_feat)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                
        return np.array(features)

def prepare_mimii_data(mimii_dir, test_size=0.2, random_state=42):
    """
    Loads MIMII data. Since acoustic anomaly detection is often framed as an 
    Unsupervised One-Class classification problem, we focus heavily on normal data.
    """
    extractor = AudioFeatureExtractor()
    
    print("Extracting Normal MIMII Features...")
    normal_path = os.path.join(mimii_dir, "**/normal")
    x_normal = extractor.process_directory(normal_path)
    
    print("Extracting Abnormal MIMII Features...")
    abnormal_path = os.path.join(mimii_dir, "**/abnormal")
    x_abnormal = extractor.process_directory(abnormal_path)
    
    # Create Labels: 0 for Normal, 1 for Abnormal
    y_normal = np.zeros(len(x_normal))
    y_abnormal = np.ones(len(x_abnormal))
    
    X = np.concatenate([x_normal, x_abnormal], axis=0)
    y = np.concatenate([y_normal, y_abnormal], axis=0)
    
    # Add a channel dimension for CNN compatibility (Batch, Time, Features, 1)
    X = np.expand_dims(X, axis=-1)
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def prepare_self_recorded_data(self_recorded_dir):
    """Loads target desk fan data (assumed completely normal for fine-tuning baseline)."""
    extractor = AudioFeatureExtractor()
    print("Extracting Self-Recorded Normal Features...")
    X = extractor.process_directory(self_recorded_dir)
    X = np.expand_dims(X, axis=-1)
    y = np.zeros(len(X)) 
    return X, y

if __name__ == "__main__":
    # Quick sanity check execution
    print("Dataset module ready.")