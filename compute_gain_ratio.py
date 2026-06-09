import numpy as np
import librosa
import glob

with open("arduino_capture.txt") as f:
    lines = f.readlines()

start = next(i for i, l in enumerate(lines) if "BEGIN_PCM" in l) + 1
end   = next(i for i, l in enumerate(lines) if "END_PCM" in l)
arduino_pcm = np.array([int(l.strip()) for l in lines[start:end]], dtype=np.float32)

# Normalize int16 to float [-1, 1] to match librosa
arduino_audio = arduino_pcm / 32768.0
arduino_rms = np.sqrt(np.mean(arduino_audio ** 2))
print(f"Arduino RMS: {arduino_rms:.6f}")

# Measure average RMS across MIMII normal files
mimii_files = glob.glob("data/fan/**/normal/*.wav", recursive=True)[:50] # sample 50
mimii_rms_values = []
for path in mimii_files:
    audio, _ = librosa.load(path, sr=16000)
    mimii_rms_values.append(np.sqrt(np.mean(audio ** 2)))

mimii_rms = np.mean(mimii_rms_values)
print(f"MIMII mean RMS: {mimii_rms:.6f}")

gain_ratio = arduino_rms / mimii_rms
print(f"Gain ratio (arduino/mimii): {gain_ratio:.6f}")
print(f"Apply this to MIMII audio before feature extraction")