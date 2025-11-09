import osimport os"""Create a tiny synthetic dataset of WAV files for quick testing.

import numpy as np

import soundfile as sfimport numpy as np

from src.main import prepare_sample_data

import soundfile as sfCreates `out_dir/real` and `out_dir/fake` each containing several short WAVs.

def make_sample_dataset():

    """Create a small sample dataset of fake/real audio pairs"""from src.main import prepare_sample_data"""

    

    # Create directoriesimport argparse

    os.makedirs("sample_dataset/fake", exist_ok=True)

    os.makedirs("sample_dataset/real", exist_ok=True)def make_sample_dataset():import os

    

    # Generate sample audio    """Create a small sample dataset of fake/real audio pairs"""import numpy as np

    duration = 3  # seconds

    sample_rate = 16000    import soundfile as sf

    t = np.linspace(0, duration, int(duration * sample_rate))

        # Create directories

    for i in range(8):

        # Generate "real" audio - clean sine wave    os.makedirs("sample_dataset/fake", exist_ok=True)

        freq = 220 * (1 + i/8)  # Varying frequency

        real_audio = np.sin(2 * np.pi * freq * t)    os.makedirs("sample_dataset/real", exist_ok=True)def make_tone(freq, duration=0.5, sr=22050):

        

        # Add some noise and harmonics to make it more realistic        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        real_audio += 0.3 * np.sin(4 * np.pi * freq * t)  # First harmonic

        real_audio += 0.1 * np.random.randn(len(t))  # Noise    # Generate sample audio    return 0.1 * np.sin(2 * np.pi * freq * t).astype('float32')

        real_audio = real_audio / np.max(np.abs(real_audio))  # Normalize

            duration = 3  # seconds

        # Generate "fake" audio - distorted version

        fake_audio = real_audio.copy()    sample_rate = 16000

        # Add more harmonics and noise

        fake_audio += 0.5 * np.sin(6 * np.pi * freq * t)  # More harmonics    t = np.linspace(0, duration, int(duration * sample_rate))def main():

        fake_audio += 0.3 * np.sin(8 * np.pi * freq * t)

        fake_audio += 0.2 * np.random.randn(len(t))  # More noise        parser = argparse.ArgumentParser()

        # Add some digital artifacts

        fake_audio[::4] *= 1.2  # Regular distortion    for i in range(8):    parser.add_argument('--out', default='sample_dataset', help='Output folder')

        fake_audio = fake_audio / np.max(np.abs(fake_audio))  # Normalize

                # Generate "real" audio - clean sine wave    parser.add_argument('--count', type=int, default=6, help='Files per class')

        # Save files

        sf.write(f"sample_dataset/real/real_{i}.wav", real_audio, sample_rate)        freq = 220 * (1 + i/8)  # Varying frequency    args = parser.parse_args()

        sf.write(f"sample_dataset/fake/fake_{i}.wav", fake_audio, sample_rate)

            real_audio = np.sin(2 * np.pi * freq * t)

    print("Sample dataset created successfully!")

            out = args.out

if __name__ == "__main__":

    make_sample_dataset()        # Add some noise and harmonics to make it more realistic    real_dir = os.path.join(out, 'real')

        real_audio += 0.3 * np.sin(4 * np.pi * freq * t)  # First harmonic    fake_dir = os.path.join(out, 'fake')

        real_audio += 0.1 * np.random.randn(len(t))  # Noise    os.makedirs(real_dir, exist_ok=True)

        real_audio = real_audio / np.max(np.abs(real_audio))  # Normalize    os.makedirs(fake_dir, exist_ok=True)

        

        # Generate "fake" audio - distorted version    sr = 22050

        fake_audio = real_audio.copy()    # real: low-frequency tones

        # Add more harmonics and noise    for i in range(args.count):

        fake_audio += 0.5 * np.sin(6 * np.pi * freq * t)  # More harmonics        y = make_tone(200 + i * 10, duration=0.7, sr=sr)

        fake_audio += 0.3 * np.sin(8 * np.pi * freq * t)        sf.write(os.path.join(real_dir, f'real_{i}.wav'), y, sr)

        fake_audio += 0.2 * np.random.randn(len(t))  # More noise

        # Add some digital artifacts    # fake: higher-frequency or chirp tones

        fake_audio[::4] *= 1.2  # Regular distortion    for i in range(args.count):

        fake_audio = fake_audio / np.max(np.abs(fake_audio))  # Normalize        y = make_tone(800 + i * 30, duration=0.7, sr=sr)

                sf.write(os.path.join(fake_dir, f'fake_{i}.wav'), y, sr)

        # Save files

        sf.write(f"sample_dataset/real/real_{i}.wav", real_audio, sample_rate)    print(f"Created synthetic dataset at {out} (real: {real_dir}, fake: {fake_dir})")

        sf.write(f"sample_dataset/fake/fake_{i}.wav", fake_audio, sample_rate)

    

    print("Sample dataset created successfully!")if __name__ == '__main__':

    main()

if __name__ == "__main__":
    make_sample_dataset()