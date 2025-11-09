# Voice Deepfake Detection with Genetic Algorithm Optimization# Voice Deepfake Detection with Genetic Algorithm Optimization# Deepfake Voice Detection (baseline + bio-inspired optimizer)



This project implements a voice deepfake detection system that is optimized using genetic algorithms. The system can detect artificially synthesized or manipulated voice recordings by analyzing audio features and patterns.



## FeaturesThis project implements a voice deepfake detection system that is optimized using genetic algorithms. The system can detect artificially synthesized or manipulated voice recordings by analyzing audio features and patterns.This project provides a small Python pipeline to:



- Audio feature extraction and analysis- extract audio features (MFCCs),

- Deep neural network-based classification

- Genetic algorithm optimization of model architecture and hyperparameters## Features- train a baseline classifier (Random Forest),

- Sample dataset for testing

- Pretrained model checkpoints- optimize hyperparameters using a simple Genetic Algorithm (bio-inspired),



## Getting Started- Audio feature extraction and analysis- and predict whether an input audio sample is a deepfake or bona fide.



1. Create a virtual environment:- Deep neural network-based classification

```bash

python -m venv .venv- Genetic algorithm optimization of model architecture and hyperparametersKey assumptions:

source .venv/bin/activate  # On Linux/Mac

.venv\Scripts\activate     # On Windows- Sample dataset for testing- You provide a labeled dataset in WAV format under a directory with two subfolders: `real/` and `fake/` (or `bonafide/` and `spoof/`).

```

- Pretrained model checkpoints- If you don't have a dataset, the repository contains a small synthetic test that runs locally.

2. Install dependencies:

```bash

pip install -r requirements.txt

```## Getting StartedFiles:



3. Run the example:- `src/audio_utils.py` — audio loading/recording and MFCC extraction.

```bash

python src/main.py1. Create a virtual environment:- `src/models.py` — baseline classifier and GA-based hyperparameter optimizer.

```

```bash- `src/main.py` — CLI to train/optimize and predict on an audio file or a live recording.

## Project Structure

python -m venv .venv- `tests/test_pipeline.py` — minimal unit test.

- `src/`: Source code

  - `audio_utils.py`: Audio processing utilitiessource .venv/bin/activate  # On Linux/Mac- `requirements.txt` — libraries needed.

  - `models.py`: Model architectures

  - `main.py`: Main execution script.venv\Scripts\activate     # On Windows

- `scripts/`: Helper scripts

  - `make_sample_dataset.py`: Generate sample dataset```How to try quickly (Windows PowerShell):

  - `prepare_asvspoof2019.py`: ASVspoof2019 dataset preparation

  - `compare_ckpts.py`: Compare model checkpoints

- `tests/`: Unit tests

- `sample_dataset/`: Sample audio files for testing2. Install dependencies:1) Create and activate a venv, then install requirements:

  - `real/`: Real voice samples  

  - `fake/`: Deepfake voice samples```bash

- `requirements.txt`: Python dependencies

pip install -r requirements.txt```powershell

## Testing

```python -m venv .venv; .\.venv\Scripts\Activate.ps1

```bash

pytest tests/pip install -r requirements.txt

```

3. Run the example:```

## License

```bash

MIT License
python src/main.py2) Run the tests:

```

```powershell

## Project Structurepytest -q

```

- `src/`: Source code

  - `audio_utils.py`: Audio processing utilities3) Train or predict (examples):

  - `models.py`: Model architectures

  - `main.py`: Main execution script```powershell

- `scripts/`: Helper scriptspython -m src.main train --data-dir path\to\dataset --optimize --generations 10

  - `make_sample_dataset.py`: Generate sample datasetpython -m src.main predict --path path\to\sample.wav

  - `prepare_asvspoof2019.py`: ASVspoof2019 dataset preparationpython -m src.main predict --path path\to\directory\with\wavs

  - `compare_ckpts.py`: Compare model checkpoints```

- `tests/`: Unit tests

- `sample_dataset/`: Sample audio files for testingNotes and limitations:

  - `real/`: Real voice samples  - This is a demo pipeline; for production, use a proper labeled dataset (e.g., ASVspoof), stronger features, and a more robust optimizer.

  - `fake/`: Deepfake voice samples- The genetic algorithm implemented is intentionally small and educational.

- `requirements.txt`: Python dependencies

## Testing

```bash
pytest tests/
```

## License

MIT License