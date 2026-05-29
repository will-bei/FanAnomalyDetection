# FanAnomalyDetection
EE P 564 TinyML Final Project: Fan Anomaly Detection. Detects abnormal conditions for a Woozoo fan.

# Dataset
The project utilizes the MIMII Dataset for initial training/testing, and self-recorded audio from a Woozoo fan for fine-tuning. 

MIMII Dataset: https://zenodo.org/records/3384388 

# Environment
I recommend using uv for setting up the project virtual environment. Simply activate the virtual environment via

> .venv\Scripts\activate

and then run:

> uv pip install -r requirements.txt

# Training
To train the model from scratch, perform the following:

> uv run .\training\train.py
> uv run .\training\fine_tune.py
> uv run .\training\evaluate.py
