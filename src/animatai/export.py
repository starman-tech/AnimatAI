"""ONNX export: `model.onnx` (features + style vector -> frames) and `model.json` (everything else)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import load_package


class _Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, feats, style):
        return self.model.forward_style_vec(feats, style)


def export_onnx(model_path: str, out_dir: str, style: str = "generic") -> Path:
    model, pkg = load_package(model_path, "cpu")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    def realistic(n):  # features distributed like the training data, not raw N(0,1)
        return model.feat_mean + model.feat_std * torch.randn(1, n, *pkg["feat_shape"])

    feats = realistic(120)
    style_vec = model.styles.weight[pkg["style_names"].index(style)][None].detach()
    torch.onnx.export(_Wrapper(model), (feats, style_vec), str(out / "model.onnx"), input_names=["feats", "style"],
                      output_names=["frames"], dynamic_axes={"feats": {1: "time2"}, "frames": {1: "time"}},
                      opset_version=17, dynamo=False)
    (out / "model.json").write_text(json.dumps({
        "format": "animatai-onnx-v1", "schema": pkg["schema"], "channels": pkg["channels"], "fps": pkg["fps"],
        "frontend": pkg["frontend"], "style_names": pkg["style_names"],
        "style_vectors": model.styles.weight.detach().numpy().round(6).tolist(),
        "lookahead_frames": model.lookahead_frames, "context_frames": model.receptive_past_frames,
        "mean_face": pkg["extra"].get("mean_face"), "metrics": pkg["metrics"]}))

    import onnxruntime as ort

    session = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    probe = realistic(200)  # different length than the trace: checks the dynamic time axis
    got = session.run(None, {"feats": probe.numpy(), "style": style_vec.numpy()})[0]
    model.eval()  # the exporter may leave modules in a different mode
    with torch.no_grad():
        want = model.forward_style_vec(probe, style_vec).numpy()
    err = float(np.abs(got - want).max())
    print(f"exported -> {out}  (max |onnx - torch| = {err:.2e})")
    if err > 1e-3:
        raise RuntimeError(f"ONNX output does not match the torch model (max error {err:.2e})")
    return out
