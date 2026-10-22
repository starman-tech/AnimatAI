"""Audio I/O: decode anything to mono float32 at the model sample rate."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


def resample(audio: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if sr == target_sr:
        return audio.astype(np.float32, copy=False)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, target_sr)
    return resample_poly(audio, target_sr // g, sr // g).astype(np.float32)


def _load_ffmpeg(path: Path, sr: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}: {proc.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load any audio/video file as mono float32 in [-1, 1].

    libsndfile handles real WAV/FLAC/OGG; everything else (the browser recorder
    saves WebM/Opus under a .wav name) goes through ffmpeg.
    """
    path = Path(path)
    try:
        import soundfile as sf

        audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        return resample(audio.mean(axis=1), file_sr, sr)
    except Exception:
        return _load_ffmpeg(path, sr)


def save_wav(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")


def rms_envelope(audio: np.ndarray, fps: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Log-energy envelope sampled at `fps` (used for A/V sync scoring)."""
    n = int(round(len(audio) / sr * fps))
    hop = sr / fps
    win = int(hop * 2)
    padded = np.pad(audio, (win // 2, win))
    starts = (np.arange(n) * hop).astype(int)
    energy = np.array([np.sqrt(np.mean(padded[s : s + win] ** 2) + 1e-10) for s in starts])
    return np.log(energy + 1e-5).astype(np.float32)
