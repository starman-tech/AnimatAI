"""HTTP API so any voice assistant / TTS pipeline can ask for an animation.

POST /v1/animate   multipart `audio` (any format) or raw body; optional form/query fields
                   style, emotion, rig(=server-side path), format=json|legacy58
GET  /v1/info      channels, styles, emotions, fps, latency
GET  /health
"""
from __future__ import annotations

import tempfile
import threading

import numpy as np

from . import schema
from .runtime.engine import Animator
from .runtime.postprocess import EMOTIONS


def create_app(model_path: str):
    from flask import Flask, jsonify, request
    from flask_cors import CORS

    app = Flask("animatai")
    CORS(app)
    animator = Animator(model_path)
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.get("/v1/info")
    def info():
        return jsonify(schema=schema.SCHEMA_VERSION, fps=animator.fps, channels=list(schema.CHANNELS),
                       styles=animator.style_names, emotions=sorted(EMOTIONS), latency_ms=animator.latency_ms)

    @app.post("/v1/animate")
    def animate():
        params = {**request.args, **request.form}
        upload = request.files.get("audio")
        data = upload.read() if upload else request.get_data()
        if not data:
            return jsonify(error="no audio received (multipart field `audio` or raw body)"), 400
        try:
            with tempfile.NamedTemporaryFile(suffix=".audio") as tmp, lock:
                tmp.write(data)
                tmp.flush()
                animator.set_style(params.get("style", "generic"))
                animator.set_emotion(params.get("emotion"))
                frames = animator.animate_file(tmp.name)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            return jsonify(error=f"could not decode audio: {exc}"), 400
        if params.get("format") == "legacy58":
            return jsonify(fps=animator.fps, frames=np.round(schema.to_legacy58(frames), 4).tolist())
        return jsonify(schema=schema.SCHEMA_VERSION, fps=animator.fps, channels=animator.channels,
                       frames=np.round(frames, 4).tolist())

    return app


def run_server(model_path: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    create_app(model_path).run(host=host, port=port, threaded=True)
