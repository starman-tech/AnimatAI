"""Training loss and evaluation metrics."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import schema
from .config import LossConfig


class A2BLoss(torch.nn.Module):
    """Group-weighted Huber (L2 by default) on positions and on frame-to-frame velocity.

    The velocity term is what keeps lips snappy: a pure position loss regresses
    to a blurry, under-articulated mouth.
    """

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("weights", torch.from_numpy(schema.channel_weights(cfg.group_weights)))

    def forward(self, pred, target, valid, time_mask) -> tuple[torch.Tensor, dict]:
        w = (self.weights * valid).unsqueeze(1)  # (B, 1, C)
        m = time_mask.unsqueeze(-1)  # (B, T, 1)
        pos = F.smooth_l1_loss(pred, target, beta=self.cfg.huber_beta, reduction="none")
        pos = (pos * w * m).sum() / ((w * m).sum().clamp_min(1.0))
        mv = m[:, 1:] * m[:, :-1]
        vel = F.smooth_l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1],
                               beta=self.cfg.huber_beta, reduction="none")
        vel = (vel * w * mv).sum() / ((w * mv).sum().clamp_min(1.0))
        loss = pos + self.cfg.velocity_weight * vel
        return loss, {"pos": float(pos), "vel": float(vel)}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_metrics(preds: list[np.ndarray], targets: list[np.ndarray], valids: list[np.ndarray],
                    reference_mean: np.ndarray | None = None) -> dict:
    """Metrics over whole clips. `mouth_corr` is the headline lip-sync number."""
    pred, target = np.concatenate(preds), np.concatenate(targets)
    valid = np.concatenate([np.repeat(v[None], len(p), 0) for v, p in zip(valids, preds)]) > 0
    out: dict = {}
    for group, idxs in schema.GROUPS.items():
        mask = valid[:, idxs]
        err = np.abs(pred[:, idxs] - target[:, idxs])
        out[f"mae_{group}"] = float(err[mask].mean()) if mask.any() else float("nan")
    # Correlations are computed per clip, then length-weighted: pooling clips would mix each
    # speaker's resting-face offset into the score, which says nothing about lip-sync.
    def clip_corr(channels: list[int]) -> float:
        total, weight = 0.0, 0.0
        for p, t in zip(preds, targets):
            active = [i for i in channels if t[:, i].std() > 0.02]
            if active and len(t) >= 30:
                total += len(t) * float(np.mean([_pearson(p[:, i], t[:, i]) for i in active]))
                weight += len(t)
        return total / weight if weight else 0.0

    mouth = [i for i in schema.GROUPS["mouth"] if target[:, i].std() > 0.02]
    out["mouth_corr"] = clip_corr(schema.GROUPS["mouth"])
    out["jaw_corr"] = clip_corr([schema.INDEX["jawOpen"]])
    out["brow_corr"] = clip_corr(schema.GROUPS["brows"])
    mi = schema.GROUPS["mouth"]
    vel_p = np.concatenate([np.diff(p[:, mi], axis=0) for p in preds])
    vel_t = np.concatenate([np.diff(t[:, mi], axis=0) for t in targets])
    out["mouth_vel_mae"] = float(np.abs(vel_p - vel_t).mean())
    # How much of the real articulation amplitude the model reproduces (1.0 == full).
    out["mouth_amplitude_ratio"] = float(pred[:, mouth].std(0).mean() / (target[:, mouth].std(0).mean() + 1e-8))
    if reference_mean is not None:
        out["mae_mouth_meanface"] = float(np.abs(reference_mean[mi][None] - target[:, mi]).mean())
    return out
