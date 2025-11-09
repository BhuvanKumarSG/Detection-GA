import osimport os"""Compare predictions from two checkpoints across a dataset folder.

import joblib

import numpy as npimport joblib

import torch

from src.main import load_modelimport numpy as npUsage:



def compare_model_checkpoints(baseline_path, optimized_path):import torch  python scripts/compare_ckpts.py --path sample_dataset --baseline detector_baseline.ckpt --optimized detector_optimized.ckpt

    """Compare the performance metrics between baseline and optimized models"""

    from src.main import load_model"""

    # Load models

    baseline_model = load_model(baseline_path)import argparse

    optimized_model = load_model(optimized_path)

    def compare_model_checkpoints(baseline_path, optimized_path):import os

    # Set both models to eval mode

    baseline_model.eval()    """Compare the performance metrics between baseline and optimized models"""from pathlib import Path

    optimized_model.eval()

        import joblib

    # Load test data (small subset for demonstration)

    test_files = []    # Load modelsimport sys

    for i in range(4):

        test_files.extend([    baseline_model = load_model(baseline_path)from pathlib import Path as _Path

            f"sample_dataset/real/real_{i}.wav",

            f"sample_dataset/fake/fake_{i}.wav"    optimized_model = load_model(optimized_path)# ensure project root is on sys.path so `src` package is importable

        ])

        _proj_root = str(_Path(__file__).resolve().parents[1])

    # Get predictions from both models

    with torch.no_grad():    # Set both models to eval modeif _proj_root not in sys.path:

        baseline_preds = baseline_model.predict(test_files)

        optimized_preds = optimized_model.predict(test_files)    baseline_model.eval()    sys.path.insert(0, _proj_root)

    

    # True labels (alternating real/fake)    optimized_model.eval()

    true_labels = np.array([i % 2 for i in range(len(test_files))])

        from src.audio_utils import load_audio, extract_features

    # Calculate accuracy

    baseline_acc = np.mean(baseline_preds == true_labels)    # Load test data (small subset for demonstration)

    optimized_acc = np.mean(optimized_preds == true_labels)

        test_files = []

    print("Model Comparison Results:")

    print("-----------------------")    for i in range(4):def load_model(path):

    print(f"Baseline Model Accuracy: {baseline_acc:.4f}")

    print(f"Optimized Model Accuracy: {optimized_acc:.4f}")        test_files.extend([    return joblib.load(path)

    print(f"Improvement: {(optimized_acc - baseline_acc)*100:.2f}%")

                f"sample_dataset/real/real_{i}.wav",

    # Save comparison results

    comparison = {            f"sample_dataset/fake/fake_{i}.wav"

        'baseline_accuracy': baseline_acc,

        'optimized_accuracy': optimized_acc,        ])def score_model(model, feats):

        'improvement': optimized_acc - baseline_acc,

        'baseline_predictions': baseline_preds,        try:

        'optimized_predictions': optimized_preds,

        'true_labels': true_labels    # Get predictions from both models        return float(model.predict_proba(feats)[0, 1])

    }

        with torch.no_grad():    except Exception:

    joblib.dump(comparison, 'model_comparison.joblib')

    print("\nComparison results saved to model_comparison.joblib")        baseline_preds = baseline_model.predict(test_files)        return float(model.predict(feats)[0])



if __name__ == "__main__":        optimized_preds = optimized_model.predict(test_files)

    compare_model_checkpoints(

        baseline_path='detector_baseline.ckpt',    

        optimized_path='detector_optimized.ckpt'

    )    # True labels (alternating real/fake)def main():

    true_labels = np.array([i % 2 for i in range(len(test_files))])    parser = argparse.ArgumentParser()

        parser.add_argument('--path', required=True)

    # Calculate accuracy    parser.add_argument('--baseline', required=True)

    baseline_acc = np.mean(baseline_preds == true_labels)    parser.add_argument('--optimized', required=True)

    optimized_acc = np.mean(optimized_preds == true_labels)    args = parser.parse_args()

    

    print("Model Comparison Results:")    base = load_model(args.baseline)

    print("-----------------------")    opt = load_model(args.optimized)

    print(f"Baseline Model Accuracy: {baseline_acc:.4f}")

    print(f"Optimized Model Accuracy: {optimized_acc:.4f}")    files = []

    print(f"Improvement: {(optimized_acc - baseline_acc)*100:.2f}%")    p = Path(args.path)

        if p.is_dir():

    # Save comparison results        for root, _, fnames in os.walk(p):

    comparison = {            for f in fnames:

        'baseline_accuracy': baseline_acc,                if f.lower().endswith('.wav'):

        'optimized_accuracy': optimized_acc,                    files.append(Path(root) / f)

        'improvement': optimized_acc - baseline_acc,    elif p.is_file():

        'baseline_predictions': baseline_preds,        files = [p]

        'optimized_predictions': optimized_preds,    else:

        'true_labels': true_labels        print('No files found')

    }        return

    

    joblib.dump(comparison, 'model_comparison.joblib')    print(f"Comparing {len(files)} files")

    print("\nComparison results saved to model_comparison.joblib")    print(f"{'file':40} {'baseline_prob':>14} {'opt_prob':>10} {'baseline_label':>14} {'opt_label':>10}")

    for f in sorted(files):

if __name__ == "__main__":        try:

    compare_model_checkpoints(            y, sr = load_audio(str(f))

        baseline_path='detector_baseline.ckpt',            feats = extract_features(y, sr=sr)[None, ...]

        optimized_path='detector_optimized.ckpt'            b = score_model(base, feats)

    )            o = score_model(opt, feats)
            bl = 'fake' if b >= 0.5 else 'real'
            ol = 'fake' if o >= 0.5 else 'real'
            print(f"{f.name:40} {b:14.3f} {o:10.3f} {bl:14} {ol:10}")
        except Exception as e:
            print(f"{f.name:40} ERROR: {e}")


if __name__ == '__main__':
    main()
