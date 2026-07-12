#!/usr/bin/env python3
"""Host TensorFlow Lite (LiteRT) float32 YOLOX MNv4-PAFPN 320² bench.

Pairs with benchmark/netkit/src/yolox_main.cc:
  - same .tflite (benchmark/tflm/generated/yolox_mnv4_pafpn_320.tflite)
  - same 10 letterboxed COCO images
  - same methodology (10 images x 5 loops, warm_mean discards first pass)
  - times interpreter.invoke() only (no decode/NMS)
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "benchmark" / "tflm" / "generated" / "yolox_mnv4_pafpn_320.tflite"
CACHE = ROOT / "benchmark" / "tflm" / "generated" / "yolox_sample_cache"
SIZE = 320
OUT_ELEMS = 178500


def _load_interpreter(model_path: Path, *, num_threads: int, use_xnnpack: bool):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    kwargs = {
        "model_path": str(model_path),
        "num_threads": num_threads,
    }
    if not use_xnnpack:
        kwargs["experimental_op_resolver_type"] = OpResolverType.BUILTIN_REF
    return Interpreter(**kwargs)


def _load_images() -> list[tuple[str, np.ndarray]]:
    if not CACHE.is_dir():
        raise SystemExit(
            f"missing {CACHE} — run: make -C benchmark/tflm export-yolox-images"
        )
    npy_files = sorted(CACHE.glob("*.npy"))
    if len(npy_files) < 10:
        raise SystemExit(
            f"need >=10 .npy under {CACHE} — run: make -C benchmark/tflm export-yolox-images"
        )
    out = []
    for path in npy_files[:10]:
        pixels = np.load(path).astype(np.float32, copy=False)
        if pixels.shape != (SIZE, SIZE, 3):
            raise SystemExit(f"bad shape {pixels.shape} in {path}")
        out.append((path.stem, pixels))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument(
        "--no-xnnpack",
        action="store_true",
        help="Disable XNNPACK (builtin reference kernels only)",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(
            f"missing {args.model} — run: make -C benchmark/tflm export-yolox"
        )

    use_xnnpack = not args.no_xnnpack
    backend = "xnnpack" if use_xnnpack else "builtin-ref"
    images = _load_images()
    num_images = len(images)

    interp = _load_interpreter(
        args.model, num_threads=args.num_threads, use_xnnpack=use_xnnpack
    )
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_shape = tuple(int(x) for x in inp["shape"])
    # Accept NHWC 1x320x320x3 or NCHW 1x3x320x320 from onnx2tf
    if in_shape == (1, SIZE, SIZE, 3):
        layout = "nhwc"
    elif in_shape == (1, 3, SIZE, SIZE):
        layout = "nchw"
    else:
        raise SystemExit(f"unexpected input shape {in_shape}")
    out_elems = int(np.prod(out["shape"]))
    if out_elems != OUT_ELEMS:
        print(f"warning: output elems {out_elems} != {OUT_ELEMS}", file=sys.stderr)

    print("TF Lite YOLOX MNv4-PAFPN float32 benchmark")
    print("  runtime:     TensorFlow Lite / LiteRT (MPU interpreter)")
    print(f"  backend:     {backend}")
    print(f"  threads:     {args.num_threads}")
    print("  dtype:       float32")
    print(f"  model:       {args.model}")
    print(f"  model bytes: {args.model.stat().st_size}")
    print(f"  input:       {in_shape} ({layout})  outputs: {out_elems}")
    print(
        f"  method:      {num_images} images x {args.loops} loops = "
        f"{num_images * args.loops} invokes (all timed)"
    )
    print("  note:        times invoke() only (no decode/NMS)")

    samples: list[float] = []
    for loop in range(args.loops):
        for i, (short, pixels) in enumerate(images):
            if layout == "nhwc":
                batch = pixels.reshape(1, SIZE, SIZE, 3)
            else:
                batch = pixels.transpose(2, 0, 1)[None, ...]
            batch = batch.astype(np.float32, copy=False)
            interp.set_tensor(inp["index"], batch)
            t0 = time.perf_counter()
            interp.invoke()
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1e6)
            if loop == 0:
                print(f"  image {i} {short}")

    cold_us = samples[0]
    first_pass_mean = statistics.fmean(samples[:num_images])
    warm = samples[num_images:]
    warm_mean = statistics.fmean(warm)
    warm_median = statistics.median(warm)
    warm_min = min(warm)
    warm_max = max(warm)
    warm_std = statistics.pstdev(warm)

    print()
    print(f"TF Lite YOLOX MNv4-PAFPN summary ({backend})")
    print(
        f"  10-image mean:    {first_pass_mean:9.3f} us ({first_pass_mean / 1000.0:7.3f} ms)"
    )
    print(f"  cold invoke:      {cold_us:9.3f} us ({cold_us / 1000.0:7.3f} ms)")
    print(f"  warm median:      {warm_median:9.3f} us ({warm_median / 1000.0:7.3f} ms)")
    print(
        f"  warm mean:        {warm_mean:9.3f} us ({warm_mean / 1000.0:7.3f} ms)"
        f"  over {len(warm)} invokes"
    )
    print(f"  warm min/max:     {warm_min:9.3f} / {warm_max:.3f} us")
    print(f"  warm stddev:      {warm_std:9.3f} us")
    print(
        "BENCHMARK_SUMMARY runtime=tflite model=yolox_mnv4_pafpn dtype=float32 "
        f"backend={backend} threads={args.num_threads} "
        f"ten_image_mean_us={first_pass_mean:.3f} "
        f"warm_median_us={warm_median:.3f} warm_mean_us={warm_mean:.3f} cold_us={cold_us:.3f} "
        f"invokes={len(samples)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
