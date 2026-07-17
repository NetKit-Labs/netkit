#!/usr/bin/env python3
"""Host ONNX Runtime float32 MNIST CNN / DS-CNN bench.

Pairs with TF Lite / netkit host A/B:
  - models/mnist_cnn.onnx (NCHW) or --model for DS-CNN
  - same float digit vectors from the .nk TCAS
  - same methodology (10 runs x 10 images, discard first invoke each run)

XNNPACK ON/OFF via --no-xnnpack. Requires LiteRT-matched ORT build:
  ./tools/build_onnxruntime_litert_matched.sh
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "benchmark" / "tflm" / "tools"
DEFAULT_MODEL = ROOT / "benchmark" / "onnxruntime" / "models" / "mnist_cnn.onnx"
FLOAT_NK = ROOT / "models" / "mnist_cnn.nk"

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TOOLS))
from export_int8_test_images import (  # noqa: E402
    INPUT_SIZE,
    NUM_IMAGES,
    _one_per_digit_cases,
)
from session_util import backend_name, input_meta, make_session, output_name  # noqa: E402


def _load_float_images(nk_path: Path) -> list[tuple[str, int, np.ndarray]]:
    sys.path.insert(0, str(ROOT / "python"))
    from netkit.reader import read_test_suite

    if not nk_path.is_file():
        raise SystemExit(f"missing {nk_path}")
    suite = read_test_suite(nk_path)
    if suite is None:
        raise SystemExit(f"missing TCAS section in {nk_path}")
    cases = _one_per_digit_cases(suite.cases, num=NUM_IMAGES)
    out: list[tuple[str, int, np.ndarray]] = []
    for case in cases:
        pixels = np.asarray(case.input, dtype=np.float32).reshape(-1)
        if pixels.size != INPUT_SIZE:
            raise SystemExit(f"case {case.name} has {pixels.size} inputs")
        out.append((case.name, int(case.label), pixels))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--nk", type=Path, default=FLOAT_NK)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument(
        "--no-xnnpack",
        action="store_true",
        help="CPU EP only (no XNNPACKExecutionProvider)",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"missing {args.model}")

    use_xnnpack = not args.no_xnnpack
    backend = backend_name(use_xnnpack)
    session = make_session(
        args.model, num_threads=args.num_threads, use_xnnpack=use_xnnpack
    )
    in_name, in_shape, in_dtype = input_meta(session)
    out_name = output_name(session)
    if in_dtype != np.float32:
        raise SystemExit(f"expected float32 input, got {in_dtype}")

    images = _load_float_images(args.nk)
    num_images = len(images)
    model_tag = "cnn_dw_f32" if "cnn_dw" in args.model.name else "cnn_f32"

    print("ONNX Runtime MNIST CNN float32 benchmark")
    print("  runtime:     ONNX Runtime (host)")
    print(f"  backend:     {backend}")
    print(f"  providers:   {session.get_providers()}")
    print(f"  threads:     {args.num_threads}")
    print("  dtype:       float32")
    print(f"  model:       {args.model}")
    print(f"  model bytes: {args.model.stat().st_size}")
    print(f"  input:       {in_name} {in_shape}")
    print(f"  images:      {num_images} per run")
    print(f"  runs:        {args.runs} (discard first invoke each run)")

    run_averages: list[float] = []
    correct = 0
    for run in range(args.runs):
        run_total = 0.0
        counted = 0
        for i, (name, label, pixels) in enumerate(images):
            # ONNX fixtures are NCHW [1,1,28,28].
            batch = pixels.reshape(in_shape).astype(np.float32, copy=False)
            feeds = {in_name: batch}
            t0 = time.perf_counter()
            outs = session.run([out_name], feeds)
            t1 = time.perf_counter()
            elapsed_us = (t1 - t0) * 1e6
            if i > 0:
                run_total += elapsed_us
                counted += 1
            if run == args.runs - 1:
                logits = np.asarray(outs[0]).reshape(-1)
                pred = int(logits.argmax())
                ok = pred == label
                correct += int(ok)
                print(
                    f"  image {i} label={label} pred={pred} "
                    f"{'OK' if ok else 'MISS'} ({name})"
                )
        run_averages.append(run_total / counted)

    if len(run_averages) < 2:
        raise SystemExit("need at least 2 runs to discard cold first run")
    warm_runs = run_averages[1:]
    mean_us = float(np.mean(warm_runs))
    print()
    print(f"ONNX Runtime MNIST {model_tag} benchmark summary ({backend})")
    print(
        f"  method:      discard run 0 + first invoke each run; "
        f"mean over {len(warm_runs)} warm runs x images 1-9"
    )
    print(f"  mean:   {mean_us:8.3f} us ({mean_us / 1000.0:6.3f} ms)")
    print(f"  accuracy:    {correct}/{num_images} on final run")
    print(
        f"BENCHMARK_SUMMARY runtime=onnxruntime model={model_tag} backend={backend} "
        f"mean_us={mean_us:.3f} runs={len(warm_runs)} "
        f"top1_correct={correct} top1_total={num_images}"
    )
    return 0 if correct == num_images else 1


if __name__ == "__main__":
    raise SystemExit(main())
