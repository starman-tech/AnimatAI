"""Typed configuration, loadable from YAML with dotted CLI overrides."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    d_model: int = 256
    dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 1, 2, 4, 8])
    # The first `symmetric_blocks` blocks look into the future (sum of their
    # dilations = lookahead in frames); the rest are causal => real-time capable.
    symmetric_blocks: int = 2
    kernel: int = 3
    dropout: float = 0.15
    feature_dropout: float = 0.1
    style_dim: int = 32


@dataclass
class LossConfig:
    group_weights: dict[str, float] = field(
        default_factory=lambda: {"mouth": 1.0, "brows": 0.5, "face": 0.5, "eyes": 0.15, "head": 0.3})
    # 1.0 == plain L2 on [0,1] targets. Small values (L1-like) regress sparse blendshapes to their
    # median (~0) and flatten the mouth: measured amplitude ratio 0.18 vs 0.41 for L2.
    huber_beta: float = 1.0
    velocity_weight: float = 4.0


@dataclass
class TrainConfig:
    # The head converges fast on frozen features (val plateau after ~5 epochs, i.e. ~3 min on a laptop GPU).
    epochs: int = 30
    steps_per_epoch: int = 200
    batch_size: int = 16
    crop_sec: float = 6.0
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_steps: int = 300
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    style_dropout: float = 0.3  # prob. of training a sample under the generic style 0
    mirror_prob: float = 0.5
    patience: int = 6
    seed: int = 1234
    num_workers: int = 0
    device: str = "auto"


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    cache_dir: str = "data/cache"
    max_clip_sec: float = 30.0
    min_clip_sec: float = 2.0
    val_fraction: float = 0.12
    # Whole sources held out as an unseen-speaker test set.
    test_sources: list[str] = field(default_factory=list)
    # Second-pass cleaning (`prepare --model`): minimum correlation between a trained
    # model's mouth prediction and the tracked mouth. Rejects voice-overs and listener shots.
    min_model_agreement: float = 0.15
    min_speech_fraction: float = 0.2
    target_smoothing: int = 5  # frames; Savitzky-Golay window against tracker jitter (0 = off)
    pitch_augment: list[float] = field(default_factory=lambda: [-2.5, 2.5])  # semitones


@dataclass
class Config:
    fps: int = 30
    frontend: dict[str, Any] = field(default_factory=lambda: {"kind": "ssl"})
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _merge(obj, values: dict):
    names = {f.name for f in fields(obj)}
    for key, value in values.items():
        if key not in names:
            raise KeyError(f"unknown config key {key!r} in {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """YAML file + ``a.b.c=value`` overrides (values parsed as YAML)."""
    cfg = Config()
    if path:
        _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node: dict = {}
        cursor = node
        parts = key.split(".")
        for part in parts[:-1]:
            cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = yaml.safe_load(raw)
        _merge(cfg, node)
    cfg.frontend.setdefault("fps", cfg.fps)
    return cfg


def config_from_dict(values: dict) -> Config:
    cfg = Config()
    _merge(cfg, values)
    return cfg
