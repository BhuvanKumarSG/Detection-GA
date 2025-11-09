import numpy as np
import os
import soundfile as sf

def generate_test_audio(duration=5.0, sample_rate=22050):
    """Generate a synthetic audio signal for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate a simple signal with multiple frequencies
    frequencies = [440, 880, 1320]  # A4, A5, E6
    signal = np.zeros_like(t)
    for f in frequencies:
        signal += np.sin(2 * np.pi * f * t)
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    return signal, sample_rate

def main():
    # Create directories if they don't exist
    for dir_path in ['data/real', 'data/fake']:
        os.makedirs(dir_path, exist_ok=True)
    
    # Generate test files
    for i in range(8):
        # Generate "real" samples with clean sine waves
        signal, sr = generate_test_audio()
        sf.write(f'data/real/real_sample_{i}.wav', signal, sr)
        
        # Generate "fake" samples with added noise
        noisy_signal = signal + np.random.normal(0, 0.1, size=len(signal))
        sf.write(f'data/fake/fake_sample_{i}.wav', noisy_signal, sr)

if __name__ == '__main__':
    main()