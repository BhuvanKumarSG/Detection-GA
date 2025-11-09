import osimport osimport numpy as np

import torch

import numpy as npimport torch

from src.main import train_model, save_model

from src.audio_utils import load_audio, extract_featuresimport numpy as npfrom src.audio_utils import extract_features



def test_pipeline():from src.main import train_model, save_modelfrom src.models import BaselineDetector

    """Test the complete training pipeline"""

    # Create dummy datafrom src.audio_utils import load_audio, extract_features

    sample_rate = 16000

    duration = 1  # 1 second

    num_samples = 4

    def test_pipeline():def test_extract_features_shape():

    # Create temporary directory for test files

    os.makedirs("temp_test", exist_ok=True)    """Test the complete training pipeline"""    sr = 22050

    

    try:    # Create dummy data    t = 0.5

        # Generate dummy audio files

        train_files = []    sample_rate = 16000    y = np.sin(2 * np.pi * 220 * np.linspace(0, t, int(sr * t)))

        train_labels = []

        for i in range(num_samples):    duration = 1  # 1 second    feats = extract_features(y, sr=sr)

            # Generate random waveform

            waveform = np.random.randn(sample_rate * duration)    num_samples = 4    assert feats.ndim == 1

            

            # Save as WAV file    

            filename = os.path.join("temp_test", f"test_{i}.wav")

            with open(filename, 'wb') as f:    # Create temporary directory for test files

                f.write(b'RIFF')

                f.write((36 + len(waveform)*2).to_bytes(4, 'little'))    os.makedirs("temp_test", exist_ok=True)def test_baseline_train_predict():

                f.write(b'WAVEfmt ')

                f.write((16).to_bytes(4, 'little'))        # tiny synthetic dataset

                f.write((1).to_bytes(2, 'little'))

                f.write((1).to_bytes(2, 'little'))    try:    from sklearn.datasets import make_classification

                f.write(sample_rate.to_bytes(4, 'little'))

                f.write((sample_rate*2).to_bytes(4, 'little'))        # Generate dummy audio files

                f.write((2).to_bytes(2, 'little'))

                f.write((16).to_bytes(2, 'little'))        train_files = []    X, y = make_classification(n_samples=60, n_features=10, random_state=0)

                f.write(b'data')

                f.write((len(waveform)*2).to_bytes(4, 'little'))        train_labels = []    clf = BaselineDetector()

                for sample in waveform:

                    value = int(sample * 32767)        for i in range(num_samples):    clf.train(X, y)

                    f.write(value.to_bytes(2, 'little', signed=True))

                        # Generate random waveform    preds = clf.predict(X[:5])

            train_files.append(filename)

            train_labels.append(i % 2)  # Alternating 0/1 labels            waveform = np.random.randn(sample_rate * duration)    assert len(preds) == 5

            

        # Train model            

        model = train_model(train_files, train_labels, epochs=1)            # Save as WAV file

                    filename = os.path.join("temp_test", f"test_{i}.wav")

        # Save model            with open(filename, 'wb') as f:

        save_model(model, os.path.join("temp_test", "test_model.ckpt"))                f.write(b'RIFF')

                        f.write((36 + len(waveform)*2).to_bytes(4, 'little'))

        assert os.path.exists(os.path.join("temp_test", "test_model.ckpt"))                f.write(b'WAVEfmt ')

                        f.write((16).to_bytes(4, 'little'))

    finally:                f.write((1).to_bytes(2, 'little'))

        # Cleanup                f.write((1).to_bytes(2, 'little'))

        import shutil                f.write(sample_rate.to_bytes(4, 'little'))

        shutil.rmtree("temp_test", ignore_errors=True)                f.write((sample_rate*2).to_bytes(4, 'little'))

                f.write((2).to_bytes(2, 'little'))

if __name__ == "__main__":                f.write((16).to_bytes(2, 'little'))

    test_pipeline()                f.write(b'data')
                f.write((len(waveform)*2).to_bytes(4, 'little'))
                for sample in waveform:
                    value = int(sample * 32767)
                    f.write(value.to_bytes(2, 'little', signed=True))
            
            train_files.append(filename)
            train_labels.append(i % 2)  # Alternating 0/1 labels
            
        # Train model
        model = train_model(train_files, train_labels, epochs=1)
        
        # Save model
        save_model(model, os.path.join("temp_test", "test_model.ckpt"))
        
        assert os.path.exists(os.path.join("temp_test", "test_model.ckpt"))
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree("temp_test", ignore_errors=True)

if __name__ == "__main__":
    test_pipeline()