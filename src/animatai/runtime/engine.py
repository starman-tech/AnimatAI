"""Inference engine: whole files (`Animator.animate`) and real-time chunks (`Animator.stream`).

Accepts either a training package (``.pt``, needs torch) or an ONNX export
directory (``model.onnx`` + ``model.json``; with the mel front-end this path
needs only numpy + onnxruntime).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import schema
from ..audio import SAMPLE_RATE, load_audio, resample
from ..features import build_frontend
from .postprocess import PostProcessor
from .rig import Rig


class _TorchBackend:
    def __init__(self, path: Path, device: str):
        import torch

        from ..model import load_package

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model, pkg = load_package(path, device)
        self.meta = {"fps": pkg["fps"], "frontend": pkg["frontend"], "style_names": pkg["style_names"],
                     "lookahead_frames": self.model.lookahead_frames,
                     "context_frames": self.model.receptive_past_frames,
                     "mean_face": pkg["extra"].get("mean_face")}
        self.style_vectors = self.model.styles.weight.detach().cpu().numpy()

    def run(self, feats: np.ndarray, style_vec: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            f = torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32))[None].to(self.device)
            s = torch.from_numpy(style_vec.astype(np.float32))[None].to(self.device)
            return self.model.forward_style_vec(f, s)[0].cpu().numpy()


class _OnnxBackend:
    def __init__(self, path: Path):
        import onnxruntime as ort

        folder = path if path.is_dir() else path.parent
        self.meta = json.loads((folder / "model.json").read_text())
        self.session = ort.InferenceSession(str(folder / "model.onnx"), providers=["CPUExecutionProvider"])
        self.style_vectors = np.asarray(self.meta["style_vectors"], dtype=np.float32)

    def run(self, feats: np.ndarray, style_vec: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"feats": np.ascontiguousarray(feats, dtype=np.float32)[None],
                                       "style": style_vec.astype(np.float32)[None]})[0][0]


class Animator:
    def __init__(self, model_path: str | Path, style: str | np.ndarray = "generic", emotion: str | dict | None = None,
                 rig: str | Path | Rig | None = None, device: str = "auto", postprocess: PostProcessor | bool = True,
                 context_sec: float = 2.0):
        path = Path(model_path)
        self.backend = _OnnxBackend(path) if (path.is_dir() or path.suffix == ".onnx") else _TorchBackend(path, device)
        meta = self.backend.meta
        self.fps: int = meta["fps"]
        self.style_names: list[str] = meta["style_names"]
        self.lookahead: int = meta["lookahead_frames"]
        self.frontend = build_frontend({**meta["frontend"], **({"device": device} if meta["frontend"]["kind"] == "ssl" else {})})
        # SSL encoders want a little future audio for stable features; mel needs half a window.
        self.frontend_lookahead_sec = 0.08 if meta["frontend"]["kind"] == "ssl" else 0.0125
        self.rest = np.asarray(meta["mean_face"], dtype=np.float32) if meta.get("mean_face") else None
        self.set_style(style)
        self.post = PostProcessor(fps=self.fps) if postprocess is True else (postprocess or None)
        if self.post is not None and emotion:
            self.post.set_emotion(emotion)
        self.rig = Rig.load(rig, self.fps) if isinstance(rig, (str, Path)) else rig
        self.context_frames = int(context_sec * self.fps) // 3 * 3
        self.reset()

    # ------------------------------------------------------------------ controls

    @property
    def channels(self) -> list[str]:
        return self.rig.names if self.rig else list(schema.CHANNELS)

    @property
    def latency_ms(self) -> float:
        """Algorithmic latency of `stream` (excludes compute time and your chunk size)."""
        return 1000.0 * (self.lookahead / self.fps + self.frontend_lookahead_sec)

    def set_style(self, style: str | np.ndarray | dict[str, float]) -> None:
        """A style name, a raw style vector, or a blend like {"generic": 0.5, "matteo": 0.5}."""
        if isinstance(style, str):
            style = {style: 1.0}
        if isinstance(style, dict):
            missing = set(style) - set(self.style_names)
            if missing:
                raise ValueError(f"unknown style(s) {sorted(missing)}; available: {self.style_names}")
            total = sum(style.values())
            style = sum(w / total * self.backend.style_vectors[self.style_names.index(n)] for n, w in style.items())
        self.style_vec = np.asarray(style, dtype=np.float32)

    def set_emotion(self, emotion: str | dict | None) -> None:
        if self.post is not None:
            self.post.set_emotion(emotion)

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start = 0  # absolute sample index of _buffer[0]
        self._received = 0
        self._emitted = 0
        if self.post is not None:
            self.post.reset()
        if self.rig is not None:
            self.rig.reset()

    # ------------------------------------------------------------------ inference

    def _finish(self, raw: np.ndarray) -> np.ndarray:
        frames = self.post(raw, self.rest) if self.post is not None else raw
        return self.rig(frames) if self.rig is not None else frames

    def predict_raw(self, audio: np.ndarray) -> np.ndarray:
        """Raw network output (T, 55) for a whole utterance, no post-processing."""
        return self.backend.run(self.frontend.extract(audio), self.style_vec)

    def animate(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        self.reset()
        return self._finish(self.predict_raw(resample(np.asarray(audio, dtype=np.float32), sr)))

    def animate_file(self, path: str | Path) -> np.ndarray:
        return self.animate(load_audio(path))

    def stream(self, chunk: np.ndarray, sr: int = SAMPLE_RATE, flush: bool = False) -> np.ndarray:
        """Feed the next audio chunk (any size), get the frames that became ready (possibly none).

        Call with ``flush=True`` on the last chunk of an utterance to drain the
        lookahead, then `reset()` before the next utterance.
        """
        chunk = resample(np.asarray(chunk, dtype=np.float32).reshape(-1), sr)
        self._buffer = np.concatenate([self._buffer, chunk])
        self._received += len(chunk)
        seconds = self._received / SAMPLE_RATE
        if flush:
            ready = int(round(seconds * self.fps))
        else:
            ready = int((seconds - self.frontend_lookahead_sec) * self.fps) - self.lookahead
        if ready <= self._emitted:
            return np.zeros((0, len(self.channels)), dtype=np.float32)

        start_frame = max(0, self._emitted - self.context_frames) // 3 * 3  # 3 frames == 1600 samples exactly
        start_sample = start_frame * SAMPLE_RATE // self.fps
        window = self._buffer[start_sample - self._buffer_start:]
        raw = self.backend.run(self.frontend.extract(window), self.style_vec)
        lo, hi = self._emitted - start_frame, min(ready - start_frame, len(raw))
        self._emitted = start_frame + hi
        # Forget audio older than the next window's context.
        keep_from = max(0, self._emitted - self.context_frames) // 3 * 3 * SAMPLE_RATE // self.fps
        self._buffer = self._buffer[keep_from - self._buffer_start:]
        self._buffer_start = keep_from
        return self._finish(raw[lo:hi])


def write_animation(path: str | Path, frames: np.ndarray, animator: Animator) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npy":
        np.save(path, frames)
    elif path.suffix == ".csv":
        np.savetxt(path, frames, delimiter=",", header=",".join(animator.channels), comments="", fmt="%.4f")
    else:
        path.write_text(json.dumps({"schema": schema.SCHEMA_VERSION, "fps": animator.fps,
                                    "channels": animator.channels, "frames": np.round(frames, 4).tolist()}))
