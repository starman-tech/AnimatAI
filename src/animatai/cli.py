"""`animatai` command line. Heavy imports are deferred to the sub-commands."""
from __future__ import annotations

import argparse
from pathlib import Path


def _config(args):
    from .config import load_config

    return load_config(args.config, args.set)


def cmd_extract(args):
    from .data.mining import extract_video

    source = args.name or Path(args.video).stem
    n = extract_video(args.video, Path(args.out) / source, source=source, speaker=args.speaker or source,
                      landmarker=args.landmarker)
    print(f"{n} clip(s) -> {Path(args.out) / source}")


def cmd_mine(args):
    from .data import mining

    tracker = Path(args.tracker)
    if args.add:
        print(f"{mining.add_links(tracker, ' '.join(args.add))} new video(s) queued")
    if not args.no_run:
        mining.mine_queue(tracker, Path(args.out), Path(args.landmarker), args.max_minutes, args.workers)


def cmd_prepare(args):
    from .data.prepare import prepare

    prepare(_config(args), model_path=args.model)


def cmd_train(args):
    from .train import train

    train(_config(args), args.run)


def cmd_finetune(args):
    from .finetune import finetune

    finetune(args.model, args.data, args.name, args.out, epochs=args.epochs, lr=args.lr, mode=args.mode)


def cmd_evaluate(args):
    from .evaluate import evaluate_package

    evaluate_package(args.model, _config(args), style=args.style, streaming=args.streaming)


def cmd_infer(args):
    from .runtime.engine import Animator, write_animation

    animator = Animator(args.model, style=args.style, emotion=args.emotion, rig=args.rig)
    frames = animator.animate_file(args.audio)
    write_animation(args.out, frames, animator)
    print(f"{len(frames)} frames @ {animator.fps} fps -> {args.out}")


def cmd_export(args):
    from .export import export_onnx

    export_onnx(args.model, args.out, style=args.style)


def cmd_serve(args):
    from .serve import run_server

    run_server(args.model, host=args.host, port=args.port)


def cmd_live(args):
    from .runtime.live import run_live

    run_live(args.model, style=args.style, emotion=args.emotion, rig=args.rig, osc=args.osc, udp=args.udp)


def cmd_demo(args):
    from .viewer import build_demo

    out = build_demo(args.model, args.out, audio_files=args.audio, processed_dir=args.processed, split=args.split,
                     n=args.n, style=args.style, emotion=args.emotion)
    print(f"demo -> {out}  (open it in a browser)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="animatai", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
        return p

    def with_config(p):
        p.add_argument("--config", default="configs/base.yaml")
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override, e.g. train.epochs=50")
        return p

    p = add("extract", cmd_extract, "video file -> training clips (use this to record a mimic)")
    p.add_argument("video")
    p.add_argument("--name", help="source name (default: file name)")
    p.add_argument("--speaker")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--landmarker", default="assets/face_landmarker.task")

    p = add("mine", cmd_mine, "download + extract the YouTube queue")
    p.add_argument("--add", nargs="*", help="YouTube links to queue")
    p.add_argument("--no-run", action="store_true")
    p.add_argument("--tracker", default="video_manager.json")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--landmarker", default="assets/face_landmarker.task")
    p.add_argument("--max-minutes", type=float, default=20.0)
    p.add_argument("--workers", type=int, default=1, help="videos processed in parallel")

    p = with_config(add("prepare", cmd_prepare, "clean, filter and split the raw clips"))
    p.add_argument("--model", default=None, help="trained package used to fix A/V lag and reject voice-overs")

    p = with_config(add("train", cmd_train, "train a base model"))
    p.add_argument("--run", default="runs/base")

    p = add("finetune", cmd_finetune, "teach a trained model one person's mimics")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="folder of raw clips (from `animatai extract`)")
    p.add_argument("--name", required=True, help="name of the new style")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--mode", choices=["style", "adapter", "full"], default="adapter")

    p = with_config(add("evaluate", cmd_evaluate, "metrics of a model package on val/test"))
    p.add_argument("--model", required=True)
    p.add_argument("--style", default="generic")
    p.add_argument("--streaming", action="store_true", help="evaluate through the chunked real-time engine")

    p = add("infer", cmd_infer, "audio file -> animation (json / csv / npy)")
    p.add_argument("--model", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--style", default="generic")
    p.add_argument("--emotion", default=None, help="e.g. happy or happy:0.6")
    p.add_argument("--rig", default=None, help="robot rig YAML: output actuator values instead of blendshapes")

    p = add("export", cmd_export, "ONNX export for torch-free deployment")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--style", default="generic")

    p = add("serve", cmd_serve, "HTTP API for voice assistants")
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)

    p = add("live", cmd_live, "microphone -> OSC / UDP in real time")
    p.add_argument("--model", required=True)
    p.add_argument("--style", default="generic")
    p.add_argument("--emotion", default=None)
    p.add_argument("--rig", default=None)
    p.add_argument("--osc", default=None, metavar="HOST:PORT")
    p.add_argument("--udp", default=None, metavar="HOST:PORT")

    p = add("demo", cmd_demo, "self-contained HTML page: animated face, model vs tracker")
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="demo.html")
    p.add_argument("--audio", nargs="*", help="your own audio files (otherwise clips of --split are used)")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--style", default="generic")
    p.add_argument("--emotion", default=None)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
