# Voice Deepfake Detection System# Deepfake Voice Detection (baseline + bio-inspired optimizer)



This project implements a voice deepfake detection system using two different models:This project provides a small Python pipeline to:

1. A baseline CNN model- extract audio features (MFCCs),

2. A GA-optimized model with enhanced architecture- train a baseline classifier (Random Forest),

- optimize hyperparameters using a simple Genetic Algorithm (bio-inspired),

## Project Structure- and predict whether an input audio sample is a deepfake or bona fide.



```Key assumptions:

detection/- You provide a labeled dataset in WAV format under a directory with two subfolders: `real/` and `fake/` (or `bonafide/` and `spoof/`).

├── data/- If you don't have a dataset, the repository contains a small synthetic test that runs locally.

│   ├── real/      # Real voice samples

│   └── fake/      # Deepfake voice samplesFiles:

├── models/        # Saved model checkpoints- `src/audio_utils.py` — audio loading/recording and MFCC extraction.

├── src/- `src/models.py` — baseline classifier and GA-based hyperparameter optimizer.

│   ├── audio_utils.py   # Audio processing utilities- `src/main.py` — CLI to train/optimize and predict on an audio file or a live recording.

│   ├── models.py        # Model definitions- `tests/test_pipeline.py` — minimal unit test.

│   └── detector.py      # Main detector class- `requirements.txt` — libraries needed.

├── scripts/      # Additional utility scripts

├── main.py      # Command-line interfaceHow to try quickly (Windows PowerShell):

├── environment.yml  # Conda environment specification

└── README.md1) Create and activate a venv, then install requirements:

```

```powershell

## Setuppython -m venv .venv; .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

1. Create conda environment:```

```bash

conda env create -f environment.yml2) Run the tests:

conda activate detection

``````powershell

pytest -q

2. Prepare your dataset:```

   - Place real voice samples in `data/real/`

   - Place fake voice samples in `data/fake/`3) Train or predict (examples):



## Usage```powershell

python -m src.main train --data-dir path\to\dataset --optimize --generations 10

### Trainingpython -m src.main predict --path path\to\sample.wav

python -m src.main predict --path path\to\directory\with\wavs

To train both models:```



```bashNotes and limitations:

python main.py --mode train --epochs 50- This is a demo pipeline; for production, use a proper labeled dataset (e.g., ASVspoof), stronger features, and a more robust optimizer.

```- The genetic algorithm implemented is intentionally small and educational.


### Prediction

For a single audio file:
```bash
python main.py --mode predict --input path/to/audio/file.wav
```

For all audio files in a directory:
```bash
python main.py --mode predict --input path/to/audio/directory
```

## Models

### Baseline Model
- Simple CNN architecture
- Standard audio feature extraction
- Basic hyperparameters

### GA-Optimized Model
- Enhanced architecture with additional layers
- Bidirectional LSTM layers
- Optimized hyperparameters
- Batch normalization for better training

## Features

- MFCC feature extraction with delta and delta-delta
- Feature normalization
- Model checkpointing
- Support for both single file and directory-based prediction
- Comparative analysis between baseline and GA-optimized models

## Requirements

All dependencies are listed in `environment.yml`. Key dependencies:
- Python 3.8
- TensorFlow
- Librosa
- NumPy
- scikit-learn