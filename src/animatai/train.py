"""Base-model training."""
from __future__ import annotations

import copy
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data.dataset import ClipDataset, build_cache, feature_stats, target_mean
from .data.prepare import load_manifest
from .features import build_frontend
from .losses import A2BLoss, compute_metrics
from .model import A2BModel, save_package

GENERIC_STYLE = "generic"


def pick_device(name: str) -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name


@torch.no_grad()
def evaluate(model: A2BModel, dataset: ClipDataset, device: str, style_id: int = 0,
             reference_mean: np.ndarray | None = None) -> dict:
    """Whole-clip metrics under one style (the generic style by default, i.e. what a new user gets)."""
    model.eval()
    preds, targets, valids = [], [], []
    for i in range(len(dataset.items)):
        sample = dataset[i]
        style = torch.tensor([style_id], device=device)
        preds.append(model(sample["feats"][None].to(device), style)[0].cpu().numpy())
        targets.append(sample["target"].numpy())
        valids.append(sample["valid"].numpy())
    return compute_metrics(preds, targets, valids, reference_mean)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        decay = min(self.decay, (1 + step) / (10 + step))
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(p.detach(), 1 - decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


def train(cfg: Config, run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = pick_device(cfg.train.device)

    frontend = build_frontend(cfg.frontend)
    feature_dir = build_cache(cfg.data.processed_dir, cfg.data.cache_dir, frontend, cfg.data.pitch_augment)
    feat_shape = frontend.feat_shape
    frontend_cfg = frontend.config()
    del frontend
    if device == "cuda":
        torch.cuda.empty_cache()

    manifest = load_manifest(cfg.data.processed_dir)
    speakers = sorted({i["speaker"] for i in manifest if i["split"] == "train"})
    style_names = [GENERIC_STYLE] + speakers
    style_ids = {name: i for i, name in enumerate(style_names)}

    is_mel = frontend_cfg["kind"] == "mel"
    train_set = ClipDataset(cfg.data.processed_dir, feature_dir, "train", style_ids, cfg.fps,
                            crop_sec=cfg.train.crop_sec, pitch_augment=cfg.data.pitch_augment,
                            mirror_prob=cfg.train.mirror_prob, mel_augment=is_mel,
                            samples_per_epoch=cfg.train.steps_per_epoch * cfg.train.batch_size)
    eval_sets = {s: ClipDataset(cfg.data.processed_dir, feature_dir, s, style_ids, cfg.fps) for s in ("val", "test")}
    eval_sets = {s: d for s, d in eval_sets.items() if d.items}
    if "val" not in eval_sets:
        raise SystemExit("no validation clips: run `animatai prepare` with more data")
    loader = DataLoader(train_set, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, drop_last=True)

    model = A2BModel(feat_shape, cfg.model, n_styles=len(style_names)).to(device)
    model.set_feature_stats(*[t.to(device) for t in feature_stats(train_set)])
    mean_face = target_mean(train_set)
    model.init_output_bias(mean_face.to(device))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M params | lookahead {model.lookahead_frames} frames "
          f"({model.lookahead_frames / cfg.fps * 1000:.0f} ms) | styles {style_names} | device {device}")

    criterion = A2BLoss(cfg.loss).to(device)
    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and "styles" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1 or "styles" in n]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.train.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=cfg.train.lr)
    total_steps = cfg.train.epochs * cfg.train.steps_per_epoch
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(
        (s + 1) / cfg.train.warmup_steps,
        0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))) * 0.98 + 0.02))
    ema = EMA(model, cfg.train.ema_decay)

    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    log_file = open(run_dir / "log.csv", "w", newline="")
    log = None
    best, best_epoch, step = -1.0, 0, 0
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        t0, sums = time.time(), {"loss": 0.0, "pos": 0.0, "vel": 0.0}
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            style = batch["style"].clone()
            style[torch.rand(style.shape, device=device) < cfg.train.style_dropout] = 0
            pred = model(batch["feats"], style)
            loss, parts = criterion(pred, batch["target"], batch["valid"], batch["time_mask"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            schedule.step()
            step += 1
            ema.update(model, step)
            sums["loss"] += float(loss)
            sums["pos"] += parts["pos"]
            sums["vel"] += parts["vel"]

        metrics = evaluate(ema.shadow, eval_sets["val"], device, reference_mean=mean_face.numpy())
        row = {"epoch": epoch, "lr": schedule.get_last_lr()[0], "sec": round(time.time() - t0, 1),
               **{f"train_{k}": v / len(loader) for k, v in sums.items()}, **{f"val_{k}": v for k, v in metrics.items()}}
        if log is None:
            log = csv.DictWriter(log_file, fieldnames=list(row))
            log.writeheader()
        log.writerow(row)
        log_file.flush()
        score = metrics["mouth_corr"]
        flag = ""
        if score > best:
            best, best_epoch, flag = score, epoch, " *"
            all_metrics = {"val": metrics}
            if "test" in eval_sets:
                all_metrics["test_unseen_speaker"] = evaluate(ema.shadow, eval_sets["test"], device,
                                                              reference_mean=mean_face.numpy())
            save_package(run_dir / "best.pt", ema.shadow, frontend_cfg, style_names, cfg.fps, all_metrics,
                         extra={"epoch": epoch, "mean_face": mean_face.tolist(), "config": cfg.to_dict()})
        print(f"ep {epoch:3d} | loss {row['train_loss']:.4f} | val mouth_corr {score:.3f} jaw {metrics['jaw_corr']:.3f} "
              f"mae_mouth {metrics['mae_mouth']:.4f} amp {metrics['mouth_amplitude_ratio']:.2f} | {row['sec']}s{flag}", flush=True)
        if epoch - best_epoch >= cfg.train.patience:
            print(f"early stop: no improvement for {cfg.train.patience} epochs")
            break
    log_file.close()
    final = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["metrics"]
    (run_dir / "metrics.json").write_text(json.dumps({"best_epoch": best_epoch, **final}, indent=2))
    print(json.dumps({"best_epoch": best_epoch, **final}, indent=2))
    return final
