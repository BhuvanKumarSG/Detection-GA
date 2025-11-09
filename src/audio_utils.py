import librosa
import numpy as np
from typing import Tuple, List


class AudioProcessor:
    def __init__(self, sample_rate: int = 22050, duration: float = 5.0, n_mfcc: int = 13):
        """
        Audio processor that returns fixed-length feature vectors suitable for
        scikit-learn models (mean/std of MFCC, delta, and delta-delta).
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            return audio, sr
        except Exception as e:
            raise Exception(f"Error loading audio file {file_path}: {str(e)}")

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features and return a fixed-length 1D vector by computing
        statistics across time (mean and std for each coefficient).
        """
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            delta = librosa.feature.delta(mfccs)
            delta2 = librosa.feature.delta(mfccs, order=2)

            # Stack -> shape (3*n_mfcc, time)
            stacked = np.vstack([mfccs, delta, delta2])

            # Compute mean and std across time for each coefficient -> fixed length
            mean_feats = np.mean(stacked, axis=1)
            std_feats = np.std(stacked, axis=1)

            features = np.concatenate([mean_feats, std_feats])

            # Safe normalization (avoid division by zero)
            if np.std(features) > 0:
                features = (features - np.mean(features)) / np.std(features)

            return features
        except Exception as e:
            raise Exception(f"Error extracting features: {str(e)}")

    def process_audio_file(self, file_path: str) -> np.ndarray:
        audio, _ = self.load_audio(file_path)
        return self.extract_features(audio)

    def process_audio_files(self, file_paths: List[str]) -> List[np.ndarray]:
        features_list = []
        for file_path in file_paths:
            features_list.append(self.process_audio_file(file_path))
        return features_list