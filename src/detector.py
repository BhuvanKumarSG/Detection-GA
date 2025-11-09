import os
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import json

from src.audio_utils import AudioProcessor
from src.models import BaselineModel, GAOptimizedModel

class DeepfakeDetector:
    def __init__(self, data_dir: str = 'data', models_dir: str = 'models'):
        """
        Initialize the Deepfake Detector.
        
        Args:
            data_dir (str): Directory containing real and fake voice samples
            models_dir (str): Directory to save/load models
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.audio_processor = AudioProcessor()
        
        # Ensure model directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize models (will be set up properly during training)
        self.baseline_model = None
        self.ga_model = None
        self.input_shape = None
        
    def _load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and process the dataset from the data directory.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels
        """
        # Get file paths
        real_dir = os.path.join(self.data_dir, 'real')
        fake_dir = os.path.join(self.data_dir, 'fake')

        real_files = []
        fake_files = []

        if os.path.isdir(real_dir):
            real_files = [os.path.join(real_dir, f)
                          for f in os.listdir(real_dir)
                          if f.endswith(('.wav', '.mp3'))]

        if os.path.isdir(fake_dir):
            fake_files = [os.path.join(fake_dir, f)
                          for f in os.listdir(fake_dir)
                          if f.endswith(('.wav', '.mp3'))]

        # Process audio files -> each item is a 1D fixed-length vector
        real_features = self.audio_processor.process_audio_files(real_files)
        fake_features = self.audio_processor.process_audio_files(fake_files)

        # Create features and labels arrays (2D X, 1D y)
        if len(real_features) + len(fake_features) == 0:
            raise Exception('No audio files found in data/real or data/fake')

        X = np.vstack(real_features + fake_features)
        y = np.array([1] * len(real_features) + [0] * len(fake_features))

        return X, y
    
    def train_models(self, test_size: float = 0.2, epochs: int = 50,
                     ga_generations: int = 12, ga_population: int = 20,
                     ga_cv: int = 3, ga_scoring: str = 'roc_auc') -> Dict:
        """
        Train both baseline and GA-optimized models.
        
        Args:
            test_size (float): Proportion of data to use for testing
            epochs (int): Number of training epochs
            
        Returns:
            Dict: Training history for both models
        """
        # Load dataset
        X, y = self._load_dataset()

        # Train baseline RandomForest
        self.baseline_model = BaselineModel(model_path=os.path.join(self.models_dir, 'baseline.ckpt'))
        self.baseline_model.train(X, y)

        # Train GA-optimized RandomForest (optimize hyperparams and save)
        self.ga_model = GAOptimizedModel(model_path=os.path.join(self.models_dir, 'ga_optimized.ckpt'))
        best_params = self.ga_model.optimize(X, y,
                                            generations=ga_generations,
                                            population_size=ga_population,
                                            cv=ga_cv,
                                            scoring=ga_scoring)

        return {
            'ga_best_params': best_params
        }
    
    def load_models(self) -> bool:
        """
        Load pre-trained models if they exist.
        
        Returns:
            bool: True if both models were loaded successfully
        """
        # Initialize models and try to load
        self.baseline_model = BaselineModel(model_path=os.path.join(self.models_dir, 'baseline.ckpt'))
        self.ga_model = GAOptimizedModel(model_path=os.path.join(self.models_dir, 'ga_optimized.ckpt'))

        baseline_loaded = self.baseline_model.load_model()
        ga_loaded = self.ga_model.load_model()
        return (baseline_loaded and ga_loaded)
    
    def predict(self, audio_path: str) -> Dict[str, float]:
        """
        Make predictions using both models.
        
        Args:
            audio_path (str): Path to the audio file to analyze
            
        Returns:
            Dict[str, float]: Prediction probabilities from both models
        """
        if not self.baseline_model or not self.ga_model:
            if not self.load_models():
                raise Exception("Models not trained. Please train the models first.")
        
        # Process audio file
        features = self.audio_processor.process_audio_file(audio_path)
        features = np.expand_dims(features, axis=0)  # shape (1, n_features)

        # Get probabilities
        baseline_prob = float(self.baseline_model.predict_proba(features)[0])
        ga_prob = float(self.ga_model.predict_proba(features)[0])

        return {
            'baseline_probability': baseline_prob,
            'ga_probability': ga_prob,
            'baseline_prediction': 'real' if baseline_prob > 0.5 else 'fake',
            'ga_prediction': 'real' if ga_prob > 0.5 else 'fake'
        }
    
    def predict_directory(self, directory: str) -> Dict[str, Dict[str, float]]:
        """
        Make predictions for all audio files in a directory.
        
        Args:
            directory (str): Path to directory containing audio files
            
        Returns:
            Dict[str, Dict[str, float]]: Predictions for each file
        """
        results = {}

        # If directory contains subfolders like 'real' and 'fake', walk them
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if not filename.endswith(('.wav', '.mp3')):
                    continue
                file_path = os.path.join(root, filename)
                try:
                    pred = self.predict(file_path)

                    # Determine true label from parent directory name if possible
                    parent = os.path.basename(os.path.normpath(root)).lower()
                    if parent == 'real':
                        true_label = 1
                    elif parent == 'fake':
                        true_label = 0
                    else:
                        true_label = None

                    baseline_prob = pred.get('baseline_probability')
                    ga_prob = pred.get('ga_probability')

                    info = dict(pred)

                    if true_label is None:
                        info['true_label'] = None
                        info['better_model'] = 'unknown'
                    else:
                        info['true_label'] = 'real' if true_label == 1 else 'fake'
                        baseline_error = abs(baseline_prob - true_label)
                        ga_error = abs(ga_prob - true_label)
                        info['baseline_correct'] = (baseline_error < 0.5)
                        info['ga_correct'] = (ga_error < 0.5)
                        info['baseline_error'] = float(baseline_error)
                        info['ga_error'] = float(ga_error)

                        if baseline_error < ga_error:
                            info['better_model'] = 'baseline'
                        elif ga_error < baseline_error:
                            info['better_model'] = 'ga_optimized'
                        else:
                            info['better_model'] = 'tie'

                    results[os.path.relpath(file_path, directory)] = info
                except Exception as e:
                    results[os.path.relpath(file_path, directory)] = {'error': str(e)}

        return results