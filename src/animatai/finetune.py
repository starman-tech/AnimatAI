"""Mimic fine-tuning: teach a trained model one person's (or character's) way of moving.

Workflow:
    animatai extract me_talking.mp4 --name matteo --out data/mimics      # a few minutes is enough
    animatai finetune --model runs/base/best.pt --data data/mimics/matteo --name matteo --out models/matteo.pt
    animatai infer --model models/matteo.pt --style matteo ...

Modes, from least to most data-hungry:
    style    only the new 32-float style vector is learned (30 s+ of video)
    adapter  style vector + FiLM generators + output head     (2 min+, default)
    full     every weight, low learning rate                   (10 min+)

The base styles are left untouched in `style`/`adapter` mode... almost: the
FiLM generators and head are shared, so `adapter` keeps them close to the base
weights with an L2 anchor instead of freezing them.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from .config import config_from_dict
from .data.dataset import ClipDataset, build_cache
from .data.prepare import load_manifest, prepare
from .features import build_frontend
from .losses import A2BLoss
from .model import load_package, save_package
from .train import evaluate, pick_device

ANCHOR_WEIGHT = 50.0


def _trainable(model, mode: str) -> list[torch.nn.Parameter]:
    for p in model.parameters():
        p.requires_grad_(mode == "full")
    model.styles.weight.requires_grad_(True)
    if mode == "adapter":
        for block in model.blocks:
            for p in block.film.parameters():
                p.requires_grad_(True)
        for p in model.head.parameters():
            p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def finetune(model_path: str, data_dir: str, name: str, out_path: str, epochs: int = 40, lr: float = 2e-4,
             mode: str = "adapter", steps_per_epoch: int = 50, device: str = "auto") -> dict:
    device = pick_device(device)
    model, pkg = load_package(model_path, device)
    cfg = config_from_dict(pkg["extra"]["config"])
    if name in pkg["style_names"]:
        raise SystemExit(f"style {name!r} already exists in {model_path}")

    work = Path(tempfile.mkdtemp(prefix="animatai_ft_"))
    cfg.data.raw_dir, cfg.data.processed_dir, cfg.data.cache_dir = str(data_dir), str(work / "processed"), str(work / "cache")
    cfg.data.test_sources = []
    cfg.data.val_fraction = 0.15
    report = prepare(cfg, verbose=False)
    minutes = report["splits"]["train"]["minutes"]
    print(f"mimic data: {minutes:.1f} min train, {report['splits']['val']['minutes']:.1f} min val "
          f"(rejected: { {s: v['rejected'] for s, v in report['sources'].items()} })")
    if minutes < 0.3:
        raise SystemExit("not enough usable footage (need at least ~30 s of clear, in-sync talking)")

    frontend = build_frontend(pkg["frontend"])
    feature_dir = build_cache(cfg.data.processed_dir, cfg.data.cache_dir, frontend, cfg.data.pitch_augment)
    del frontend

    style_id = model.add_style(init_from=0)
    style_names = pkg["style_names"] + [name]
    ids = {speaker: style_id for speaker in {i["speaker"] for i in load_manifest(cfg.data.processed_dir)}}
    is_mel = pkg["frontend"]["kind"] == "mel"
    train_set = ClipDataset(cfg.data.processed_dir, feature_dir, "train", ids, cfg.fps, crop_sec=cfg.train.crop_sec,
                            pitch_augment=cfg.data.pitch_augment, mel_augment=is_mel,
                            samples_per_epoch=steps_per_epoch * cfg.train.batch_size)
    val_set = ClipDataset(cfg.data.processed_dir, feature_dir, "val", ids, cfg.fps)
    if not val_set.items:  # tiny dataset: monitor on train clips rather than not at all
        val_set = ClipDataset(cfg.data.processed_dir, feature_dir, "train", ids, cfg.fps)
    loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.train.batch_size, drop_last=True)

    before_generic = evaluate(model, val_set, device, style_id=0)
    params = _trainable(model, mode)
    anchors = [p.detach().clone() for p in params]
    style_rows = model.styles.weight.detach().clone()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    criterion = A2BLoss(cfg.loss).to(device)
    print(f"mode={mode}: {sum(p.numel() for p in params):,} trainable values")

    best_state, best_score = None, -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        for m in model.modules():  # small data: keep dropout off, only the chosen weights move
            if isinstance(m, torch.nn.Dropout):
                m.eval()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = criterion(model(batch["feats"], batch["style"]), batch["target"], batch["valid"], batch["time_mask"])
            if mode != "style":
                loss = loss + ANCHOR_WEIGHT * sum(((p - a) ** 2).mean() for p, a in zip(params, anchors)
                                                  if p is not model.styles.weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():  # existing styles are frozen: only the new row learns
                model.styles.weight.grad[:style_id] = 0
            optimizer.step()
            with torch.no_grad():
                model.styles.weight[:style_id] = style_rows[:style_id]
        metrics = evaluate(model, val_set, device, style_id=style_id)
        score = metrics["mouth_corr"] - metrics["mae_mouth"] - metrics["mae_brows"]
        if score > best_score:
            best_score, best_metrics = score, metrics
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"ep {epoch:3d} | mouth_corr {metrics['mouth_corr']:.3f} mae_mouth {metrics['mae_mouth']:.4f} "
                  f"mae_brows {metrics['mae_brows']:.4f}")

    model.load_state_dict(best_state)
    summary = {"style": name, "mode": mode, "train_minutes": minutes,
               "before_generic_style": before_generic, "after_mimic_style": best_metrics}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_package(out_path, model, pkg["frontend"], style_names, pkg["fps"],
                 metrics={**pkg["metrics"], f"mimic_{name}": summary},
                 extra={**pkg["extra"], "mimic_mean_face": {**pkg["extra"].get("mimic_mean_face", {}),
                                                             name: np.concatenate(train_set.targets).mean(0).tolist()}})
    print(json.dumps({k: (v if not isinstance(v, dict) else {m: round(x, 4) for m, x in v.items()})
                      for k, v in summary.items()}, indent=2))
    print(f"saved -> {out_path}  (use --style {name})")
    return summary
