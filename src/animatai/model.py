"""A2B network: audio features -> 52 blendshapes + head rotation, per frame.

A small temporal conv net over frozen front-end features. Almost all blocks are
causal so the same weights serve offline files and low-latency streaming; every
block is modulated (FiLM) by a *style* vector, which is what a "mimic"
fine-tune learns.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import schema
from .config import ModelConfig


class FiLMBlock(nn.Module):
    def __init__(self, d: int, kernel: int, dilation: int, symmetric: bool, style_dim: int, dropout: float):
        super().__init__()
        span = (kernel - 1) * dilation
        self.pad = (span // 2, span - span // 2) if symmetric else (span, 0)
        self.lookahead = self.pad[1]
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, 2 * d, kernel, dilation=dilation)
        self.film = nn.Linear(style_dim, 2 * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2)
        h = F.glu(self.conv(F.pad(h, self.pad)), dim=1).transpose(1, 2)
        gamma, beta = self.film(style).unsqueeze(1).chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta
        return x + self.drop(self.out(h))


class A2BModel(nn.Module):
    def __init__(self, feat_shape: tuple[int, ...], cfg: ModelConfig, n_styles: int = 1):
        super().__init__()
        self.cfg = cfg
        self.feat_shape = tuple(feat_shape)
        feat_dim = self.feat_shape[-1]
        self.register_buffer("feat_mean", torch.zeros(self.feat_shape))
        self.register_buffer("feat_std", torch.ones(self.feat_shape))
        # SSL front-ends give several encoder layers: learn how to mix them.
        self.layer_logits = nn.Parameter(torch.zeros(self.feat_shape[0])) if len(self.feat_shape) == 2 else None
        self.feat_drop = nn.Dropout(cfg.feature_dropout)
        self.in_proj = nn.Linear(2 * feat_dim, cfg.d_model)  # stride-2: two feature frames per output frame
        self.styles = nn.Embedding(n_styles, cfg.style_dim)
        nn.init.normal_(self.styles.weight, std=0.02)
        self.blocks = nn.ModuleList(
            FiLMBlock(cfg.d_model, cfg.kernel, dil, i < cfg.symmetric_blocks, cfg.style_dim, cfg.dropout)
            for i, dil in enumerate(cfg.dilations))
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, schema.N_CHANNELS)

    @property
    def lookahead_frames(self) -> int:
        return sum(b.lookahead for b in self.blocks)

    @property
    def receptive_past_frames(self) -> int:
        return sum(b.pad[0] for b in self.blocks)

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std.clamp_min(1e-3))

    def init_output_bias(self, target_mean: torch.Tensor) -> None:
        """Start from the dataset's mean face instead of sigmoid(0)=0.5 everywhere."""
        with torch.no_grad():
            bs = target_mean[: schema.N_BLENDSHAPES].clamp(0.01, 0.99)
            self.head.bias[: schema.N_BLENDSHAPES] = torch.log(bs / (1 - bs))
            self.head.bias[schema.N_BLENDSHAPES:] = torch.atanh(target_mean[schema.N_BLENDSHAPES:].clamp(-0.9, 0.9))

    def add_style(self, init_from: int = 0) -> int:
        """Append a new style slot (used by mimic fine-tuning). Returns its id."""
        old = self.styles
        self.styles = nn.Embedding(old.num_embeddings + 1, old.embedding_dim).to(old.weight.device)
        with torch.no_grad():
            self.styles.weight[:-1] = old.weight
            self.styles.weight[-1] = old.weight[init_from]
        return old.num_embeddings

    def forward_style_vec(self, feats: torch.Tensor, style_vec: torch.Tensor) -> torch.Tensor:
        """feats (B, 2T, *feat_shape), style_vec (B, style_dim) -> (B, T, 55)."""
        x = (feats.float() - self.feat_mean) / self.feat_std
        if self.layer_logits is not None:
            x = (x * torch.softmax(self.layer_logits, 0).view(1, 1, -1, 1)).sum(dim=2)
        x = self.feat_drop(x)
        x = self.in_proj(x.reshape(x.shape[0], -1, 2 * x.shape[-1]))  # -1 keeps the ONNX time axis dynamic
        for block in self.blocks:
            x = block(x, style_vec)
        y = self.head(self.out_norm(x))
        return torch.cat([torch.sigmoid(y[..., : schema.N_BLENDSHAPES]),
                          torch.tanh(y[..., schema.N_BLENDSHAPES:])], dim=-1)

    def forward(self, feats: torch.Tensor, style_id: torch.Tensor | None = None) -> torch.Tensor:
        if style_id is None:
            style_id = torch.zeros(feats.shape[0], dtype=torch.long, device=feats.device)
        return self.forward_style_vec(feats, self.styles(style_id))


# --------------------------------------------------------------------------- packaging


def save_package(path, model: A2BModel, frontend_cfg: dict, style_names: list[str], fps: int,
                 metrics: dict | None = None, extra: dict | None = None) -> None:
    """One self-describing file: weights + everything needed to run them."""
    from dataclasses import asdict

    torch.save({
        "format": "animatai-package-v1",
        "schema": schema.SCHEMA_VERSION,
        "channels": list(schema.CHANNELS),
        "fps": fps,
        "frontend": frontend_cfg,
        "feat_shape": list(model.feat_shape),
        "model_cfg": asdict(model.cfg),
        "style_names": list(style_names),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metrics": metrics or {},
        "extra": extra or {},
    }, path)


def load_package(path, device: str = "cpu") -> tuple[A2BModel, dict]:
    pkg = torch.load(path, map_location="cpu", weights_only=False)
    if pkg.get("schema") != schema.SCHEMA_VERSION:
        raise ValueError(f"{path}: schema {pkg.get('schema')!r} != {schema.SCHEMA_VERSION!r}")
    model = A2BModel(tuple(pkg["feat_shape"]), ModelConfig(**pkg["model_cfg"]), n_styles=len(pkg["style_names"]))
    model.load_state_dict(pkg["state_dict"])
    return model.to(device).eval(), pkg
