import numpy as np
import pytest
import torch

from animatai import schema
from animatai.audio import SAMPLE_RATE
from animatai.config import ModelConfig, load_config
from animatai.data.prepare import lagged_corr
from animatai.features import MelFrontend, n_output_frames
from animatai.losses import A2BLoss, compute_metrics
from animatai.config import LossConfig
from animatai.model import A2BModel, load_package, save_package
from animatai.runtime.postprocess import PostProcessor, emotion_vector, parse_emotion
from animatai.runtime.rig import Rig


def test_schema_layout():
    assert schema.N_CHANNELS == 55 and len(set(schema.CHANNELS)) == 55
    grouped = sorted(i for idxs in schema.GROUPS.values() for i in idxs)
    assert grouped == list(range(1, 55))  # everything but `_neutral`, no overlap


def test_mirror_is_involution():
    x = np.random.rand(7, 55).astype(np.float32)
    assert np.allclose(schema.mirror(schema.mirror(x)), x)
    assert not np.allclose(schema.mirror(x), x)


def test_legacy_roundtrip():
    legacy = np.random.rand(5, 58).astype(np.float32)
    frames, valid = schema.from_legacy58(legacy)
    assert frames.shape == (5, 55) and valid[schema.INDEX["headPitch"]] == 0
    assert np.allclose(schema.to_legacy58(frames)[:, 6:], legacy[:, 6:], atol=1e-6)


def test_mel_frontend_shapes():
    fe = MelFrontend(fps=30)
    for seconds in (0.5, 1.0, 2.37):
        audio = np.random.randn(int(seconds * SAMPLE_RATE)).astype(np.float32) * 0.1
        feats = fe.extract(audio)
        assert feats.shape == (2 * n_output_frames(len(audio), 30), 80)
        assert np.isfinite(feats).all()


def _model(feat_shape=(80,)):
    torch.manual_seed(0)
    return A2BModel(feat_shape, ModelConfig(d_model=32, dilations=[1, 2, 4], symmetric_blocks=1), n_styles=2).eval()


def test_model_shapes_and_ranges():
    for shape in ((80,), (3, 16)):
        out = _model(shape)(torch.randn(2, 60, *shape))
        assert out.shape == (2, 30, 55)
        assert out[..., :52].min() >= 0 and out[..., :52].max() <= 1 and out[..., 52:].abs().max() <= 1


def test_model_lookahead_is_bounded():
    """Frames must not depend on audio beyond the declared lookahead: that is what makes streaming valid."""
    model = _model()
    a = torch.randn(1, 120, 80)
    b = a.clone()
    cut = 40  # output frame index
    b[:, 2 * (cut + model.lookahead_frames + 1):] += 5.0
    ya, yb = model(a), model(b)
    assert torch.allclose(ya[:, : cut + 1], yb[:, : cut + 1], atol=1e-5)
    assert not torch.allclose(ya[:, -1], yb[:, -1], atol=1e-3)


def test_package_roundtrip_and_add_style(tmp_path):
    model = _model()
    save_package(tmp_path / "m.pt", model, {"kind": "mel", "fps": 30}, ["generic", "a"], 30)
    loaded, pkg = load_package(tmp_path / "m.pt")
    x = torch.randn(1, 40, 80)
    assert torch.allclose(model(x), loaded(x))
    new_id = loaded.add_style()
    assert new_id == 2
    assert torch.allclose(loaded(x, torch.tensor([2])), loaded(x, torch.tensor([0])))


def test_loss_ignores_invalid_channels_and_padding():
    crit = A2BLoss(LossConfig())
    pred, target = torch.rand(2, 20, 55), torch.rand(2, 20, 55)
    valid, mask = torch.ones(2, 55), torch.ones(2, 20)
    valid[:, schema.INDEX["headPitch"]] = 0
    mask[:, 15:] = 0
    base, _ = crit(pred, target, valid, mask)
    pred2 = pred.clone()
    pred2[:, :, schema.INDEX["headPitch"]] += 3.0
    pred2[:, 16:] += 3.0
    assert torch.allclose(base, crit(pred2, target, valid, mask)[0])


def test_metrics_perfect_prediction():
    t = [np.random.rand(100, 55).astype(np.float32)]
    m = compute_metrics(t, t, [np.ones(55, np.float32)])
    assert m["mouth_corr"] == pytest.approx(1.0) and m["mae_mouth"] == 0.0


def test_lagged_corr_finds_video_delay():
    rng = np.random.default_rng(0)
    env = rng.standard_normal(500)
    mouth = np.roll(env, 3)  # video 3 frames late
    scores = {lag: lagged_corr(env, mouth, lag) for lag in range(-5, 6)}
    assert max(scores, key=scores.get) == 3


def test_postprocess_blinks_emotion_and_clipping():
    post = PostProcessor(fps=30, seed=0)
    post.set_emotion("happy:0.8")
    out = post(np.zeros((300, 55), np.float32))
    assert out[:, schema.BLINK[0]].max() > 0.8  # blinked within 10 s
    assert out[-1, schema.INDEX["mouthSmileLeft"]] == pytest.approx(0.8 * 0.55, abs=0.02)
    assert out[:, :52].min() >= 0 and out[:, :52].max() <= 1
    assert parse_emotion("happy:0.5,sad") == {"happy": 0.5, "sad": 1.0}
    with pytest.raises(ValueError):
        parse_emotion("hungry")
    assert emotion_vector({}).sum() == 0


def test_rig_mapping_and_slew_limit(tmp_path):
    rig_file = tmp_path / "rig.yaml"
    rig_file.write_text("actuators:\n  jaw:\n    channels: {jawOpen: 1.0}\n    input: [0, 1]\n    output: [90, 150]\n"
                        "    max_speed: 300\n  lid:\n    channels: {eyeBlinkLeft: 1.0}\n    output: [100, 40]\n")
    rig = Rig.load(rig_file, fps=30)
    frames = np.zeros((3, 55), np.float32)
    frames[1:, schema.INDEX["jawOpen"]] = 1.0
    frames[1:, schema.INDEX["eyeBlinkLeft"]] = 1.0
    out = rig(frames)
    assert out[0].tolist() == [90, 100]
    assert out[1, 0] == pytest.approx(100)  # 300 deg/s at 30 fps == 10 deg per frame
    assert out[1, 1] == 40  # unlimited actuator, reversed range


def test_config_overrides(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("fps: 30\nmodel:\n  d_model: 64\n")
    cfg = load_config(cfg_file, ["train.epochs=3", "frontend.kind=mel"])
    assert cfg.model.d_model == 64 and cfg.train.epochs == 3 and cfg.frontend == {"kind": "mel", "fps": 30}
    with pytest.raises(KeyError):
        load_config(cfg_file, ["train.nope=1"])
