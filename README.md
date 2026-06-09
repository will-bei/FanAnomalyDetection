# FanAnomalyDetection
EE P 564 TinyML Final Project: Fan Anomaly Detection. Detects abnormal conditions for a Woozoo fan.

This repo contains two pipelines; one for an Acoustic Classifier and one for an autoencoder. The classifier unfortunately fails to recognize anomalies and only instead defaults to "Normal" whenever it hears the fan it was fine-tuned on. Thus, the rest of the README instructions are written for use with the autoencoder pipeline.

# Dataset
The project utilizes the MIMII Dataset for initial training/testing, and self-recorded audio from a Woozoo fan for fine-tuning. 

MIMII Dataset: https://zenodo.org/records/3384388 

# Environment
I recommend using uv for setting up the project virtual environment. Simply activate the virtual environment via

> .venv\Scripts\activate

and then run:

> uv pip install -r requirements.txt

# Prerequisites
An audio sample from the Arduino microphone is required to calibrate the training data for that device; MiMII dataset is recorded using professional (better) microphones. simply comment out the main loop in .\deployment\anomaly_detector_autoencoder.ino and uncomment the loop code that records a sample using the Arduino's onboard microphone. Then, export the serial monitor output to arduino_capture.txt and run compute_gain_ratio.py. Use the output of that python script to set the correct gain_correction in AudioFeatureExtractor class in dataset_autoencoder.py.

# Training
To train the model from scratch, perform the following:

> uv run .\run_autoencoder_pipeline.py

# Arduino deployment
Make sure that Harvard TinyML library is installed and that tflite-micro-arduino is installed (github link: https://github.com/tensorflow/tflite-micro-arduino-examples). After pipeline is run and the libraries are installed, run the sketch.