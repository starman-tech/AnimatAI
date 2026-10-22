"""End-to-end on synthetic clips: prepare -> train -> finetune -> export -> ONNX runtime."""
import json

import numpy as np

from animatai import schema
from animatai.audio import SAMPLE_RATE, save_wav
from animatai.config import load_config
from animatai.data.prepare import _denoise, model_agreement, prepare


def _write_clip(folder, source, index, seconds=8.0, fps=30.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    # A slow "syllable" envelope drives both the loudness and the jaw: learnable on purpose.
    env = np.clip(np.convolve(rng.random(n), np.ones(5) / 5, mode="same") * 1.5 - 0.3, 0, 1)
    frames = np.zeros((n, schema.N_CHANNELS), np.float32)
    frames[:, schema.INDEX["jawOpen"]] = env
    frames[:, schema.INDEX["browInnerUp"]] = 0.2
    n_samples = int(seconds * SAMPLE_RATE)
    audio = rng.standard_normal(n_samples) * np.interp(np.arange(n_samples) / SAMPLE_RATE, np.arange(n) / fps, env)
    folder.mkdir(parents=True, exist_ok=True)
    clip = f"{source}_{index:08d}"
    save_wav(folder / f"{clip}.wav", 0.3 * audio.astype(np.float32))
    np.savez_compressed(folder / f"{clip}.npz", frames=frames, valid=np.ones(schema.N_CHANNELS, np.float32))
    (folder / f"{clip}.meta.json").write_text(json.dumps(
        {"schema": schema.SCHEMA_VERSION, "fps": fps, "source": source, "speaker": source, "start_sec": index * 10.0}))


def test_denoise_and_agreement():
    rng = np.random.default_rng(0)
    clean = np.clip(np.cumsum(rng.standard_normal((300, 55)) * 0.02, 0) + 0.5, 0, 1).astype(np.float32)
    noisy = np.clip(clean + rng.standard_normal(clean.shape).astype(np.float32) * 0.03, 0, 1)
    assert np.abs(_denoise(noisy, 5) - clean).mean() < np.abs(noisy - clean).mean()
    assert model_agreement(clean, clean) > 0.99
    assert model_agreement(np.roll(clean, 3, axis=0), clean, lag=-3) > 0.95
    assert abs(model_agreement(rng.random(clean.shape), clean)) < 0.2


def test_full_pipeline(tmp_path):
    raw = tmp_path / "raw"
    for s, source in enumerate(["spk_a", "spk_b", "spk_held_out"]):
        for k in range(5):
            _write_clip(raw / source, source, k, seed=10 * s + k)
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("frontend: {kind: mel}\nmodel: {d_model: 32, dilations: [1, 2, 4], symmetric_blocks: 1}\n")
    overrides = [f"data.raw_dir={raw}", f"data.processed_dir={tmp_path / 'proc'}", f"data.cache_dir={tmp_path / 'cache'}",
                 "data.test_sources=[spk_held_out]", "data.pitch_augment=[]", "data.val_fraction=0.25",
                 "train.epochs=2", "train.steps_per_epoch=3", "train.batch_size=2", "train.crop_sec=2.0",
                 "train.device=cpu", "train.warmup_steps=2"]
    cfg = load_config(cfg_file, overrides)
    report = prepare(cfg, verbose=False)
    assert report["splits"]["test"]["chunks"] == 5 and report["splits"]["val"]["chunks"] >= 1

    from animatai.train import train

    metrics = train(cfg, tmp_path / "run")
    assert {"val", "test_unseen_speaker"} <= set(metrics)

    # second-pass cleaning with the trained model must run and keep the manifest usable
    prepare(cfg, verbose=False, model_path=str(tmp_path / "run" / "best.pt"))

    from animatai.finetune import finetune

    summary = finetune(str(tmp_path / "run" / "best.pt"), str(raw / "spk_held_out"), "mimic", str(tmp_path / "mimic.pt"),
                       epochs=1, steps_per_epoch=2, device="cpu")
    assert summary["style"] == "mimic"

    from animatai.export import export_onnx
    from animatai.runtime.engine import Animator

    export_onnx(str(tmp_path / "mimic.pt"), str(tmp_path / "onnx"), style="mimic")
    audio = np.random.default_rng(0).standard_normal(SAMPLE_RATE * 2).astype(np.float32) * 0.1
    torch_out = Animator(tmp_path / "mimic.pt", style="mimic", postprocess=False, device="cpu").predict_raw(audio)
    onnx_out = Animator(tmp_path / "onnx", style="mimic", postprocess=False).predict_raw(audio)
    assert torch_out.shape == (60, 55) and np.abs(torch_out - onnx_out).max() < 1e-4
