import argparse
import os
import json
from src.detector import DeepfakeDetector

def main():
    parser = argparse.ArgumentParser(description='Voice Deepfake Detection System')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                       help='Operation mode: train models or predict on new data')
    parser.add_argument('--input', type=str,
                       help='Path to audio file or directory for prediction')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (for train mode)')
    parser.add_argument('--ga-generations', type=int, default=12,
                       help='GA generations (for GA optimizer)')
    parser.add_argument('--ga-population', type=int, default=20,
                       help='GA population size (for GA optimizer)')
    parser.add_argument('--ga-cv', type=int, default=3,
                       help='Cross-validation folds for GA evaluation')
    parser.add_argument('--ga-scoring', type=str, default='roc_auc',
                       help='Scoring metric for GA evaluation (e.g., roc_auc, accuracy, f1)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = DeepfakeDetector()
    
    if args.mode == 'train':
        print("Training baseline RandomForest and GA-optimized RandomForest...")
        result = detector.train_models(epochs=args.epochs,
                                       ga_generations=args.ga_generations,
                                       ga_population=args.ga_population,
                                       ga_cv=args.ga_cv,
                                       ga_scoring=args.ga_scoring)
        print("Training completed.")
        if 'ga_best_params' in result:
            print(f"GA best params: {result['ga_best_params']}")
        
    elif args.mode == 'predict':
        if not args.input:
            print("Error: --input path is required for predict mode")
            return
        
        if os.path.isfile(args.input):
            # Single file prediction
            results = detector.predict(args.input)
            print(f"\nResults for {args.input}:")
            print(f"Baseline Model: {results['baseline_prediction']} "
                  f"(Confidence: {results['baseline_probability']:.2%})")
            print(f"GA-Optimized Model: {results['ga_prediction']} "
                  f"(Confidence: {results['ga_probability']:.2%})")
            
        elif os.path.isdir(args.input):
            # Directory prediction (will also compare model performance per sample)
            results = detector.predict_directory(args.input)
            print(f"\nResults for directory {args.input}:")

            summary = {'baseline_better': 0, 'ga_better': 0, 'tie': 0, 'unknown': 0}

            for filename, result in sorted(results.items()):
                if 'error' in result:
                    print(f"\n{filename}: Error - {result['error']}")
                    continue

                print(f"\n{filename}:")
                print(f"  Baseline: {result['baseline_prediction']} (Conf: {result['baseline_probability']:.2%})")
                print(f"  GA-Optimized: {result['ga_prediction']} (Conf: {result['ga_probability']:.2%})")

                better = result.get('better_model', 'unknown')
                if better == 'baseline':
                    summary['baseline_better'] += 1
                    print("  Better: Baseline")
                elif better == 'ga_optimized':
                    summary['ga_better'] += 1
                    print("  Better: GA-Optimized")
                elif better == 'tie':
                    summary['tie'] += 1
                    print("  Better: Tie")
                else:
                    summary['unknown'] += 1
                    print("  Better: Unknown (no true label)")

            print("\nSummary:")
            print(f"  Baseline better: {summary['baseline_better']}")
            print(f"  GA-Optimized better: {summary['ga_better']}")
            print(f"  Tie: {summary['tie']}")
            print(f"  Unknown: {summary['unknown']}")
        else:
            print(f"Error: Input path {args.input} does not exist")

if __name__ == "__main__":
    main()