"""Evaluate a model package on the processed val/test splits, offline or through the streaming engine."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, load_audio
from .config import Config
from .data.prepare import load_manifest
from .losses import compute_metrics
from .runtime.engine import Animator


def evaluate_package(model_path: str, cfg: Config, style: str = "generic", streaming: bool = False,
                     chunk_ms: int = 100) -> dict:
    animator = Animator(model_path, style=style, postprocess=False)
    processed = Path(cfg.data.processed_dir)
    manifest = load_manifest(processed)
    results = {}
    for split in ("val", "test"):
        items = [i for i in manifest if i["split"] == split]
        if not items:
            continue
        preds, targets, valids = [], [], []
        for item in items:
            audio = load_audio(processed / "audio" / f"{item['id']}.wav")
            target = np.load(processed / "targets" / f"{item['id']}.npy")
            if streaming:
                animator.reset()
                step = SAMPLE_RATE * chunk_ms // 1000
                parts = [animator.stream(audio[s: s + step]) for s in range(0, len(audio), step)]
                parts.append(animator.stream(np.zeros(0, np.float32), flush=True))
                pred = np.concatenate(parts)
            else:
                pred = animator.predict_raw(audio)
            n = min(len(pred), len(target))
            preds.append(pred[:n])
            targets.append(target[:n])
            valids.append(np.asarray(item["valid"], dtype=np.float32))
        results[split] = compute_metrics(preds, targets, valids)
    results["mode"] = f"streaming ({chunk_ms} ms chunks, {animator.latency_ms:.0f} ms lookahead)" if streaming else "offline"
    print(json.dumps(results, indent=2))
    return results
