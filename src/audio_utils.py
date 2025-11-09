import numpy as npimport numpy as np"""Audio utilities: load/record audio and extract MFCC features.

import torch

import torchaudioimport torch

import soundfile as sf

import torchaudioThe functions use librosa when available and fall back to simple numpy-based features

def load_audio(file_path):

    """Load audio file and return waveform"""import soundfile as sfif librosa is not installed. Recording uses sounddevice when available.

    try:

        waveform, sample_rate = torchaudio.load(file_path)"""

        return waveform.numpy(), sample_rate

    except Exception:def load_audio(file_path):from typing import Tuple

        waveform, sample_rate = sf.read(file_path)

        if len(waveform.shape) > 1:    """Load audio file and return waveform"""import os

            waveform = waveform.mean(axis=1)  # Convert stereo to mono

        return waveform, sample_rate    try:import numpy as np



def extract_features(waveform, sample_rate):        waveform, sample_rate = torchaudio.load(file_path)

    """Extract audio features from waveform"""

    # Convert waveform to torch tensor        return waveform.numpy(), sample_ratetry:

    if not torch.is_tensor(waveform):

        waveform = torch.from_numpy(waveform).float()    except Exception:    import librosa

    if len(waveform.shape) == 1:

        waveform = waveform.unsqueeze(0)        waveform, sample_rate = sf.read(file_path)except Exception:

        

    # Extract mel spectrogram        if len(waveform.shape) > 1:    librosa = None

    mel_transform = torchaudio.transforms.MelSpectrogram(

        sample_rate=sample_rate,            waveform = waveform.mean(axis=1)  # Convert stereo to mono

        n_fft=2048,

        hop_length=512,        return waveform, sample_ratetry:

        n_mels=128

    )    import sounddevice as sd

    mel_spec = mel_transform(waveform)

    def extract_features(waveform, sample_rate):    import soundfile as sf

    # Convert to dB scale

    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)    """Extract audio features from waveform"""except Exception:

    

    # Normalize    # Convert waveform to torch tensor    sd = None

    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()

        if not torch.is_tensor(waveform):    sf = None

    return mel_spec_db

        waveform = torch.from_numpy(waveform).float()

def preprocess_batch(audio_files):

    """Preprocess a batch of audio files for model input"""    if len(waveform.shape) == 1:

    features = []

    for file in audio_files:        waveform = waveform.unsqueeze(0)def load_audio(path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:

        waveform, sr = load_audio(file)

        features.append(extract_features(waveform, sr))            """Load audio from a WAV file. Returns (y, sr).

    return torch.stack(features)
    # Extract mel spectrogram

    mel_transform = torchaudio.transforms.MelSpectrogram(    If librosa is available, uses it; otherwise tries soundfile.

        sample_rate=sample_rate,    """

        n_fft=2048,    if not os.path.exists(path):

        hop_length=512,        raise FileNotFoundError(path)

        n_mels=128    if librosa is not None:

    )        y, seen_sr = librosa.load(path, sr=sr, mono=True)

    mel_spec = mel_transform(waveform)        return y, seen_sr

        else:

    # Convert to dB scale        if sf is None:

    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)            raise RuntimeError("Neither librosa nor soundfile are available to load audio")

            y, seen_sr = sf.read(path)

    # Normalize        # ensure mono

    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()        if y.ndim > 1:

                y = np.mean(y, axis=1)

    return mel_spec_db        if seen_sr != sr:

            # naive resample using scipy if available

def preprocess_batch(audio_files):            try:

    """Preprocess a batch of audio files for model input"""                from scipy.signal import resample

    features = []

    for file in audio_files:                num = int(len(y) * sr / seen_sr)

        waveform, sr = load_audio(file)                y = resample(y, num)

        features.append(extract_features(waveform, sr))            except Exception:

    return torch.stack(features)                pass
        return y.astype(np.float32), sr


def record_audio(duration: float = 3.0, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """Record audio from default microphone for duration seconds.

    Returns (y, sr). Requires sounddevice and soundfile.
    If recording isn't possible, raises RuntimeError.
    """
    if sd is None or sf is None:
        raise RuntimeError("Recording requires sounddevice and soundfile packages")
    print(f"Recording {duration}s from default microphone...")
    y = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    y = y.flatten()
    return y, sr


def extract_features(y: np.ndarray, sr: int = 22050, n_mfcc: int = 20) -> np.ndarray:
    """Extract MFCC-based features and return a 1D feature vector.

    Features: mean and std of MFCCs and their deltas.
    """
    if librosa is not None:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feats = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(delta, axis=1), np.std(delta, axis=1),
            np.mean(delta2, axis=1), np.std(delta2, axis=1)
        ])
        return feats.astype(np.float32)
    else:
        # fallback: use very simple spectral features
        # compute short-time energy and spectral centroid-like proxy
        frame_len = int(0.025 * sr)
        hop = int(0.010 * sr)
        frames = []
        for i in range(0, max(1, len(y) - frame_len), hop):
            frame = y[i:i + frame_len]
            energy = np.sum(frame ** 2)
            spec_cent = np.sum(np.abs(np.fft.rfft(frame)) * np.arange(len(frame) // 2 + 1))
            frames.append([energy, spec_cent])
        frames = np.array(frames) if frames else np.zeros((1, 2))
        feats = np.concatenate([np.mean(frames, axis=0), np.std(frames, axis=0)])
        return feats.astype(np.float32)


if __name__ == "__main__":
    # tiny manual test when run directly
    print("audio_utils quick smoke test")
    sr = 22050
    t = 0.5
    y = np.sin(2 * np.pi * 220 * np.linspace(0, t, int(sr * t)))
    f = extract_features(y, sr=sr)
    print("features shape:", f.shape)
