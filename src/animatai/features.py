"""Audio front-ends.

Two interchangeable front-ends turn a waveform into a feature sequence sampled
at exactly ``2 * fps`` (the model's first conv has stride 2):

* ``mel``  - pure-numpy log-mel. No torch needed at runtime: this is the one
  for Raspberry-Pi class robots and the ONNX export.
* ``ssl``  - a frozen self-supervised speech encoder (default mHuBERT-147,
  multilingual). Far more speaker/language robust, needs torch + ~95M params.

Both expose ``extract(audio) -> (2*T, *feat_shape)`` with ``T = round(dur*fps)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import SAMPLE_RATE


def n_output_frames(n_samples: int, fps: int) -> int:
    return max(1, int(round(n_samples / SAMPLE_RATE * fps)))


def feature_times(t_out: int, fps: int) -> np.ndarray:
    """Sample times of the 2*T feature grid: pair (2j, 2j+1) straddles target time j/fps."""
    return (np.arange(2 * t_out) - 0.5) / (2 * fps)


def resample_time(feats: np.ndarray, src_times: np.ndarray, dst_times: np.ndarray) -> np.ndarray:
    """Linear interpolation of a (T, ...) sequence along time (edge-clamped)."""
    if len(feats) == 1:
        return np.repeat(feats, len(dst_times), axis=0)
    pos = np.interp(dst_times, src_times, np.arange(len(src_times)))
    lo = np.floor(pos).astype(int).clip(0, len(feats) - 2)
    frac = (pos - lo).astype(np.float32).reshape(-1, *([1] * (feats.ndim - 1)))
    return feats[lo] * (1.0 - frac) + feats[lo + 1] * frac


# --------------------------------------------------------------------------- mel


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sr: int, fmin: float, fmax: float) -> np.ndarray:
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = edges[i : i + 3]
        up = (freqs - lo) / (mid - lo)
        down = (hi - freqs) / (hi - mid)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


@dataclass
class MelFrontend:
    fps: int = 30
    n_mels: int = 80
    n_fft: int = 400
    hop: int = 160
    fmin: float = 50.0
    fmax: float = 7600.0
    kind: str = "mel"
    _fb: np.ndarray = field(init=False, repr=False)
    _window: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._fb = mel_filterbank(self.n_mels, self.n_fft, SAMPLE_RATE, self.fmin, self.fmax)
        self._window = np.hanning(self.n_fft + 1)[:-1].astype(np.float32)

    @property
    def tag(self) -> str:
        return f"mel{self.n_mels}_fps{self.fps}"

    @property
    def feat_shape(self) -> tuple[int, ...]:
        return (self.n_mels,)

    def config(self) -> dict:
        return {"kind": "mel", "fps": self.fps, "n_mels": self.n_mels, "n_fft": self.n_fft,
                "hop": self.hop, "fmin": self.fmin, "fmax": self.fmax}

    def logmel(self, audio: np.ndarray) -> np.ndarray:
        """(n_frames, n_mels) log-mel at SAMPLE_RATE/hop Hz, frame i centred on i*hop."""
        pad = self.n_fft // 2
        x = np.pad(audio.astype(np.float32), (pad, pad), mode="reflect" if len(audio) > pad else "constant")
        n = 1 + (len(x) - self.n_fft) // self.hop
        idx = np.arange(self.n_fft)[None, :] + self.hop * np.arange(n)[:, None]
        spec = np.abs(np.fft.rfft(x[idx] * self._window, axis=1)) ** 2
        return np.log(spec @ self._fb.T + 1e-6).astype(np.float32)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        mel = self.logmel(audio)
        t_out = n_output_frames(len(audio), self.fps)
        src = np.arange(len(mel)) * self.hop / SAMPLE_RATE
        return resample_time(mel, src, feature_times(t_out, self.fps))


# --------------------------------------------------------------------------- ssl


@dataclass
class SSLFrontend:
    fps: int = 30
    model_id: str = "utter-project/mHuBERT-147"
    layers: tuple[int, ...] = (6, 9, 12)
    device: str = "auto"
    chunk_sec: float = 20.0
    context_sec: float = 1.0
    kind: str = "ssl"
    _model: object = field(default=None, init=False, repr=False)

    @property
    def tag(self) -> str:
        name = self.model_id.split("/")[-1].lower().replace("-", "")
        return f"{name}_L{'-'.join(map(str, self.layers))}_fps{self.fps}"

    @property
    def feat_shape(self) -> tuple[int, ...]:
        self._load()
        return (len(self.layers), self._model.config.hidden_size)

    def config(self) -> dict:
        return {"kind": "ssl", "fps": self.fps, "model_id": self.model_id, "layers": list(self.layers)}

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(self.model_id).eval().to(self.device)
        # Layers above the last one we read are dead weight: drop them.
        model.encoder.layers = model.encoder.layers[: max(self.layers)]
        if self.device == "cuda":
            model = model.half()
        self._model = model

    def _forward(self, audio: np.ndarray) -> np.ndarray:
        import torch

        x = (audio - audio.mean()) / (audio.std() + 1e-7)
        if len(x) < 400:  # shorter than one conv receptive field
            x = np.pad(x, (0, 400 - len(x)))
        t = torch.from_numpy(x.astype(np.float32))[None].to(self.device)
        if self.device == "cuda":
            t = t.half()
        with torch.no_grad():
            hs = self._model(t, output_hidden_states=True).hidden_states
        return torch.stack([hs[i][0] for i in self.layers], dim=1).float().cpu().numpy()

    def raw(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Native 50 Hz features (n, L, D) + their centre times, chunked for long audio."""
        self._load()
        chunk, ctx = int(self.chunk_sec * SAMPLE_RATE), int(self.context_sec * SAMPLE_RATE)
        times = lambda n, off: (np.arange(n) * 320 + 200 + off) / SAMPLE_RATE  # noqa: E731
        if len(audio) <= chunk + 2 * ctx:
            f = self._forward(audio)
            return f, times(len(f), 0)
        feats, ts = [], []
        for start in range(0, len(audio), chunk):
            lo, hi = max(0, start - ctx), min(len(audio), start + chunk + ctx)
            f = self._forward(audio[lo:hi])
            t = times(len(f), lo)
            keep = (t >= start / SAMPLE_RATE) & (t < (start + chunk) / SAMPLE_RATE)
            feats.append(f[keep])
            ts.append(t[keep])
        return np.concatenate(feats), np.concatenate(ts)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        f, t = self.raw(audio)
        t_out = n_output_frames(len(audio), self.fps)
        return resample_time(f, t, feature_times(t_out, self.fps)).astype(np.float16)


def build_frontend(cfg: dict):
    cfg = dict(cfg)
    kind = cfg.pop("kind")
    if kind == "mel":
        return MelFrontend(**cfg)
    if kind == "ssl":
        if "layers" in cfg:
            cfg["layers"] = tuple(cfg["layers"])
        return SSLFrontend(**cfg)
    raise ValueError(f"unknown frontend kind: {kind!r}")
