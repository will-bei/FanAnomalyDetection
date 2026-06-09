import numpy as np
import librosa

audio, sr = librosa.load("data/fan/id_00/normal/00000000.wav", sr=16000)
window = audio[:16000]

D = librosa.stft(window, n_fft=1024, hop_length=1000, window='hann')
mags = np.abs(D)  # shape (513, 17)

# Sum of squared magnitudes for frame 0, bins 1-512
test_energy = np.sum(mags[1:513, 0] ** 2)
print(f"Python sum of squared mags frame 0: {test_energy:.2f}")

import numpy as np
import librosa

audio, sr = librosa.load("data/fan/id_00/normal/00000000.wav", sr=16000)
window = audio[:16000]

mel = librosa.feature.melspectrogram(
    y=window, sr=sr, n_mels=104, n_fft=1024, hop_length=1000, power=2.0
)
log_mel = librosa.power_to_db(mel, ref=np.max)  # (104, 16)

global_mean = np.load("training/global_mean.npy")
global_std  = np.load("training/global_std.npy")
normalized = (log_mel - global_mean.squeeze()) / global_std.squeeze()

print("Normalized min/max/mean/std:", 
      normalized.min(), normalized.max(), 
      normalized.mean(), normalized.std())
print("% of values in [-2, 2]:", 
      np.mean(np.abs(normalized) < 2) * 100)