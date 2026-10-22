"""Microphone -> animation in real time, sent as OSC messages and/or UDP JSON datagrams."""
from __future__ import annotations

import json
import queue
import socket

import numpy as np

from ..audio import SAMPLE_RATE
from .engine import Animator


def _endpoint(spec: str) -> tuple[str, int]:
    host, _, port = spec.rpartition(":")
    return host or "127.0.0.1", int(port)


def run_live(model_path: str, style: str = "generic", emotion: str | None = None, rig: str | None = None,
             osc: str | None = None, udp: str | None = None, chunk_ms: int = 100) -> None:
    import sounddevice as sd

    animator = Animator(model_path, style=style, emotion=emotion, rig=rig)
    names = animator.channels
    osc_client = None
    if osc:
        from pythonosc.udp_client import SimpleUDPClient

        osc_client = SimpleUDPClient(*_endpoint(osc))
    udp_sock, udp_addr = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), _endpoint(udp)) if udp else (None, None)
    if not (osc_client or udp_sock):
        print("no --osc/--udp target: printing one channel as a level meter")
    probe = names.index("jawOpen") if "jawOpen" in names else 0

    chunks: queue.Queue[np.ndarray] = queue.Queue()
    block = SAMPLE_RATE * chunk_ms // 1000
    print(f"live: {animator.fps} fps, ~{animator.latency_ms + chunk_ms:.0f} ms latency + compute. Ctrl+C to stop.")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block,
                        callback=lambda data, *_: chunks.put(data[:, 0].copy())):
        try:
            while True:
                for frame in animator.stream(chunks.get()):
                    if osc_client:
                        for name, value in zip(names, frame):
                            osc_client.send_message(f"/animatai/{name}", float(value))
                    if udp_sock:
                        udp_sock.sendto(json.dumps(dict(zip(names, np.round(frame, 4).tolist()))).encode(), udp_addr)
                    if not (osc_client or udp_sock):
                        print(f"\r{names[probe]}={frame[probe]:7.2f}", end="")
        except KeyboardInterrupt:
            print("\nstopped")
