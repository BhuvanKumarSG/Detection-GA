import osimport os"""CLI entrypoint for training/optimizing and predicting deepfake audio.

import torch

import torch.nn as nnimport torch

import torch.optim as optim

from torch.utils.data import DataLoader, TensorDatasetimport torch.nn as nnUsage:

import numpy as np

from .models import DeepfakeDetectorimport torch.optim as optim  python -m src.main train --data-dir PATH [--optimize]

from .audio_utils import preprocess_batch

from torch.utils.data import DataLoader, TensorDataset  python -m src.main predict --file sample.wav

def train_model(train_files, train_labels, val_files=None, val_labels=None, 

                epochs=10, batch_size=32, learning_rate=0.001):import numpy as np  python -m src.main record --duration 3 --predict

    """Train the deepfake detection model"""

    # Prepare datafrom .models import DeepfakeDetector"""

    X_train = preprocess_batch(train_files)

    y_train = torch.tensor(train_labels)from .audio_utils import preprocess_batchimport argparse

    

    train_dataset = TensorDataset(X_train, y_train)import os

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def train_model(train_files, train_labels, val_files=None, val_labels=None, import numpy as np

    if val_files is not None and val_labels is not None:

        X_val = preprocess_batch(val_files)                epochs=10, batch_size=32, learning_rate=0.001):

        y_val = torch.tensor(val_labels)

        val_dataset = TensorDataset(X_val, y_val)    """Train the deepfake detection model"""from .audio_utils import load_audio, record_audio, extract_features

        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Prepare datafrom .models import BaselineDetector, GeneticOptimizer

    # Initialize model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    X_train = preprocess_batch(train_files)

    model = DeepfakeDetector().to(device)

        y_train = torch.tensor(train_labels)

    # Loss and optimizer

    criterion = nn.CrossEntropyLoss()    def load_dataset(data_dir: str):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_dataset = TensorDataset(X_train, y_train)    """Load dataset from data_dir. Expect subfolders 'real' and 'fake' (or 'bonafide'/'spoof').

    # Training loop

    for epoch in range(epochs):    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.train()

        train_loss = 0        Returns X, y arrays.

        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)    if val_files is not None and val_labels is not None:    """

            optimizer.zero_grad()

            output = model(data)        X_val = preprocess_batch(val_files)    X = []

            loss = criterion(output, target)

            loss.backward()        y_val = torch.tensor(val_labels)    y = []

            optimizer.step()

            train_loss += loss.item()        val_dataset = TensorDataset(X_val, y_val)    positive_folders = ['fake', 'spoof']

            

        avg_train_loss = train_loss / len(train_loader)        val_loader = DataLoader(val_dataset, batch_size=batch_size)    negative_folders = ['real', 'bonafide']

        

        # Validation        for label, folders in enumerate([negative_folders, positive_folders]):

        if val_files is not None and val_labels is not None:

            model.eval()    # Initialize model        for folder in folders:

            val_loss = 0

            correct = 0    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")            p = os.path.join(data_dir, folder)

            total = 0

            with torch.no_grad():    model = DeepfakeDetector().to(device)            if not os.path.exists(p):

                for data, target in val_loader:

                    data, target = data.to(device), target.to(device)                    continue

                    output = model(data)

                    val_loss += criterion(output, target).item()    # Loss and optimizer            for fname in os.listdir(p):

                    pred = output.argmax(dim=1, keepdim=True)

                    correct += pred.eq(target.view_as(pred)).sum().item()    criterion = nn.CrossEntropyLoss()                if not fname.lower().endswith('.wav'):

                    total += target.size(0)

                optimizer = optim.Adam(model.parameters(), lr=learning_rate)                    continue

            avg_val_loss = val_loss / len(val_loader)

            accuracy = 100. * correct / total                    path = os.path.join(p, fname)

            print(f'Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={accuracy:.2f}%')

        else:    # Training loop                try:

            print(f'Epoch {epoch}: Train Loss={avg_train_loss:.4f}')

                for epoch in range(epochs):                    y_audio, sr = load_audio(path)

    return model

        model.train()                    feats = extract_features(y_audio, sr=sr)

def save_model(model, path):

    """Save model checkpoint"""        train_loss = 0                    X.append(feats)

    torch.save(model.state_dict(), path)

        for batch_idx, (data, target) in enumerate(train_loader):                    y.append(label)

def load_model(path):

    """Load model from checkpoint"""            data, target = data.to(device), target.to(device)                except Exception as e:

    model = DeepfakeDetector()

    model.load_state_dict(torch.load(path))            optimizer.zero_grad()                    print(f"Skipping {path}: {e}")

    return model

            output = model(data)    if not X:

def predict(model, audio_files):

    """Make predictions on audio files"""            loss = criterion(output, target)        raise RuntimeError(f"No audio files found in {data_dir}. Expected subfolders like 'real' and 'fake'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)            loss.backward()    return np.stack(X), np.array(y)

    model.eval()

                optimizer.step()

    # Preprocess audio

    features = preprocess_batch(audio_files)            train_loss += loss.item()

    features = features.to(device)

                def train_command(args):

    # Get predictions

    with torch.no_grad():        avg_train_loss = train_loss / len(train_loader)    X, y = load_dataset(args.data_dir)

        output = model(features)

        predictions = output.argmax(dim=1)            print(f"Loaded dataset: X={X.shape}, y={y.shape}")

    

    return predictions.cpu().numpy()        # Validation    # Train and optionally optimize. If optimize requested, save two checkpoints:

        if val_files is not None and val_labels is not None:    # - detector_baseline.ckpt (baseline trained with default params)

            model.eval()    # - detector_optimized.ckpt (trained with GA-optimized hyperparameters)

            val_loss = 0    if args.optimize:

            correct = 0        # Train baseline detector first and save baseline checkpoint (if not present)

            total = 0        baseline = BaselineDetector()

            with torch.no_grad():        baseline.train(X, y)

                for data, target in val_loader:        baseline_ckpt = 'detector_baseline.ckpt'

                    data, target = data.to(device), target.to(device)        try:

                    output = model(data)            import joblib

                    val_loss += criterion(output, target).item()

                    pred = output.argmax(dim=1, keepdim=True)            if os.path.exists(baseline_ckpt):

                    correct += pred.eq(target.view_as(pred)).sum().item()                print(f"Baseline checkpoint already exists at {baseline_ckpt}; not overwriting.")

                    total += target.size(0)            else:

                            joblib.dump(baseline, baseline_ckpt)

            avg_val_loss = val_loss / len(val_loader)                print(f"Saved baseline checkpoint to {baseline_ckpt}")

            accuracy = 100. * correct / total        except Exception as e:

            print(f'Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={accuracy:.2f}%')            print(f"Failed to save baseline checkpoint: {e}")

        else:

            print(f'Epoch {epoch}: Train Loss={avg_train_loss:.4f}')        # Run optimizer and train optimized detector

                    print("Optimizing hyperparameters with Genetic Algorithm...")

    return model        opt = GeneticOptimizer(pop_size=12, generations=args.generations)

        best_params, best_score = opt.optimize(X, y)

def save_model(model, path):        print("Best params:", best_params, "score:", best_score)

    """Save model checkpoint"""        optimized = BaselineDetector(best_params)

    torch.save(model.state_dict(), path)        optimized.train(X, y)

        # Save optimized model when compare_command runs optimization as well

def load_model(path):        # (compare_command will also attempt to save as detector_optimized.ckpt)

    """Load model from checkpoint"""        optimized_ckpt = 'detector_optimized.ckpt'

    model = DeepfakeDetector()        try:

    model.load_state_dict(torch.load(path))            if os.path.exists(optimized_ckpt):

    return model                print(f"Optimized checkpoint already exists at {optimized_ckpt}; not overwriting.")

            else:

def predict(model, audio_files):                joblib.dump(optimized, optimized_ckpt)

    """Make predictions on audio files"""                print(f"Saved optimized checkpoint to {optimized_ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        except Exception as e:

    model = model.to(device)            print(f"Failed to save optimized checkpoint: {e}")

    model.eval()        # also respect --save override (single path) if provided: save optimized there if not exists

            if args.save:

    # Preprocess audio            try:

    features = preprocess_batch(audio_files)                if os.path.exists(args.save):

    features = features.to(device)                    print(f"Requested save path {args.save} exists; not overwriting.")

                    else:

    # Get predictions                    joblib.dump(optimized, args.save)

    with torch.no_grad():                    print(f"Saved optimized model to {args.save}")

        output = model(features)            except Exception as e:

        predictions = output.argmax(dim=1)                print(f"Failed to save to requested path {args.save}: {e}")

        else:

    return predictions.cpu().numpy()        detector = BaselineDetector()
        detector.train(X, y)
        # save model if requested; default save path is detector.ckpt
        save_path = args.save or 'detector.ckpt'
        try:
            import joblib

            if os.path.exists(save_path):
                print(f"Checkpoint already exists at {save_path}; not overwriting.")
            else:
                joblib.dump(detector, save_path)
                print(f"Saved trained model to {save_path}")
        except Exception as e:
            print(f"Failed to save model to {save_path}: {e}")


def predict_path(args):
    """Predict on a single file or all WAV files inside a directory specified by --path."""
    # prefer provided model path; otherwise use 'detector.ckpt' if present
    model_path = args.model
    if not model_path:
        default_ckpt = 'detector.ckpt'
        if os.path.exists(default_ckpt):
            model_path = default_ckpt

    if model_path and os.path.exists(model_path):
        import joblib

        detector = joblib.load(model_path)
    else:
        # if no model available, instantiate default and warn
        print("No saved model provided or found — using untrained default model (results may be meaningless).")
        detector = BaselineDetector()

    target = args.path
    files = []
    if os.path.isdir(target):
        for fname in os.listdir(target):
            if fname.lower().endswith('.wav'):
                files.append(os.path.join(target, fname))
    elif os.path.isfile(target):
        files = [target]
    else:
        raise RuntimeError(f"Provided path does not exist: {target}")

    for path in files:
        try:
            y_audio, sr = load_audio(path)
            feats = extract_features(y_audio, sr=sr)[None, ...]
            try:
                proba = float(detector.predict_proba(feats)[0, 1])
            except Exception:
                proba = float(detector.predict(feats)[0])
            label = 'fake' if proba >= 0.5 else 'real'
            print(f"{os.path.basename(path)} -> deepfake_prob={proba:.3f} label={label}")
        except Exception as e:
            print(f"Skipping {path}: {e}")


def compare_command(args):
    """Train a GA-optimized detector on labeled --train-data and compare predictions
    from baseline and optimized models on files under --path.
    """
    # Load baseline detector (from model path or default ckpt)
    model_path = args.model
    if not model_path:
        default_ckpt = 'detector.ckpt'
        if os.path.exists(default_ckpt):
            model_path = default_ckpt

    if model_path and os.path.exists(model_path):
        import joblib

        baseline = joblib.load(model_path)
        print(f"Loaded baseline model from {model_path}")
    else:
        print("No baseline checkpoint found; using default (untrained) baseline detector")
        baseline = BaselineDetector()

    # Prepare training data for optimizer
    train_dir = args.train_data
    try:
        X_train, y_train = load_dataset(train_dir)
        print(f"Loaded training data for optimizer: X={X_train.shape}, y={y_train.shape}")
    except Exception as e:
        print(f"Failed to load training data from {train_dir}: {e}")
        return

    # Run GA optimizer to get best hyperparameters
    print(f"Running GeneticOptimizer for {args.generations} generations...")
    opt = GeneticOptimizer(pop_size=12, generations=args.generations)
    best_params, best_score = opt.optimize(X_train, y_train)
    print(f"GA best params: {best_params} (score={best_score:.3f})")

    optimized = BaselineDetector(best_params)
    optimized.train(X_train, y_train)

    # Gather target files (reuse predict_path logic)
    target = args.path
    files = []
    if os.path.isdir(target):
        for root, _, fnames in os.walk(target):
            for fname in fnames:
                if fname.lower().endswith('.wav'):
                    files.append(os.path.join(root, fname))
    elif os.path.isfile(target):
        files = [target]
    else:
        print(f"Provided path does not exist: {target}")
        return

    # For each file, print baseline and optimized predictions side-by-side
    for path in files:
        try:
            y_audio, sr = load_audio(path)
            feats = extract_features(y_audio, sr=sr)[None, ...]
            try:
                bproba = float(baseline.predict_proba(feats)[0, 1])
            except Exception:
                bproba = float(baseline.predict(feats)[0])
            try:
                oproba = float(optimized.predict_proba(feats)[0, 1])
            except Exception:
                oproba = float(optimized.predict(feats)[0])

            blabel = 'fake' if bproba >= 0.5 else 'real'
            olabel = 'fake' if oproba >= 0.5 else 'real'
            print(f"{os.path.basename(path)} -> baseline: {bproba:.3f} ({blabel}) | optimized: {oproba:.3f} ({olabel})")
        except Exception as e:
            print(f"Skipping {path}: {e}")


def record_and_predict(args):
    y_audio, sr = record_audio(duration=args.duration)
    feats = extract_features(y_audio, sr=sr)[None, ...]
    model_path = args.model
    if model_path and os.path.exists(model_path):
        import joblib

        detector = joblib.load(model_path)
    else:
        detector = BaselineDetector()
    try:
        proba = detector.predict_proba(feats)[0, 1]
    except Exception:
        pred = detector.predict(feats)[0]
        proba = float(pred)
    print(f"Deepfake probability: {proba:.3f}")
    print('fake' if proba >= 0.5 else 'real')


def main():
    parser = argparse.ArgumentParser(description='Deepfake voice detection demo')
    sub = parser.add_subparsers(dest='cmd')

    p_train = sub.add_parser('train')
    p_train.add_argument('--data-dir', required=True)
    p_train.add_argument('--optimize', action='store_true')
    p_train.add_argument('--generations', type=int, default=8)
    p_train.add_argument('--save', default='detector.joblib')

    p_pred = sub.add_parser('predict')
    p_pred.add_argument('--path', required=True, help='Path to a WAV file or a directory with WAV files')
    p_pred.add_argument('--model', default=None, help='Optional path to a saved model (joblib)')
    p_comp = sub.add_parser('compare')
    p_comp.add_argument('--path', required=True, help='Path to a WAV file or a directory with WAV files to predict')
    p_comp.add_argument('--train-data', required=False, default='sample_dataset', help='Labeled dataset folder to train/optimize on (contains real/ and fake/)')
    p_comp.add_argument('--model', default=None, help='Optional path to a saved baseline model (joblib)')
    p_comp.add_argument('--generations', type=int, default=6, help='Generations for GA optimizer')

    args = parser.parse_args()
    if args.cmd == 'train':
        train_command(args)
    elif args.cmd == 'predict':
        predict_path(args)
    elif args.cmd == 'compare':
        compare_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
