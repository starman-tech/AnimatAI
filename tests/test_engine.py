"""Streaming must reproduce offline inference: the property every real-time integration relies on."""
import numpy as np
import torch

from animatai.audio import SAMPLE_RATE
from animatai.config import ModelConfig
from animatai.model import A2BModel, save_package
from animatai.runtime.engine import Animator


def _package(tmp_path):
    torch.manual_seed(0)
    model = A2BModel((80,), ModelConfig(d_model=32, dilations=[1, 2, 4, 8], symmetric_blocks=2), n_styles=2)
    path = tmp_path / "m.pt"
    save_package(path, model, {"kind": "mel", "fps": 30}, ["generic", "other"], 30, extra={"mean_face": [0.0] * 55})
    return path


def test_streaming_matches_offline(tmp_path):
    animator = Animator(_package(tmp_path), postprocess=False, device="cpu", context_sec=3.0)
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(SAMPLE_RATE * 6) * 0.1).astype(np.float32)
    offline = animator.predict_raw(audio)
    animator.reset()
    parts = []
    for start in range(0, len(audio), 1234):  # deliberately not frame aligned
        parts.append(animator.stream(audio[start:start + 1234]))
    parts.append(animator.stream(np.zeros(0, np.float32), flush=True))
    streamed = np.concatenate(parts)
    assert len(streamed) == len(offline)
    assert all(len(p) < 10 for p in parts[:-1])  # frames come out incrementally, not all at the end
    # Identical away from the very end (flush sees a shorter future than offline did).
    assert np.abs(streamed[:-8] - offline[:-8]).max() < 2e-3


def test_styles_and_blend(tmp_path):
    animator = Animator(_package(tmp_path), postprocess=False, device="cpu")
    animator.set_style({"generic": 0.5, "other": 0.5})
    assert animator.style_vec.shape == (32,)
    try:
        animator.set_style("nobody")
    except ValueError as exc:
        assert "available" in str(exc)
    else:
        raise AssertionError("unknown style accepted")
