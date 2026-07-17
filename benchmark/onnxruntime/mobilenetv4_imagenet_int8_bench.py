#!/usr/bin/env python3
"""Host ONNX Runtime int8 (QDQ) MobileNetV4 ImageNet bench."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "benchmark" / "tflm" / "tools"
DEFAULT_MODEL = (
    ROOT / "benchmark" / "onnxruntime" / "models" / "mobilenetv4_imagenet_int8.onnx"
)
CACHE = ROOT / "benchmark" / "tflm" / "generated" / "imagenet_sample_cache"

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TOOLS))
from export_imagenet_mnv4_test_images import SAMPLES, preprocess  # noqa: E402
from session_util import backend_name, input_meta, make_session, output_name  # noqa: E402


def _load_images_nchw() -> list[tuple[str, int, np.ndarray]]:
    from PIL import Image

    out = []
    for filename, _url, label, short in SAMPLES:
        path = CACHE / filename
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        nhwc = np.asarray(preprocess(rgb), dtype=np.float32)
        if nhwc.ndim == 3:
            nhwc = nhwc[None, ...]
        nchw = np.transpose(nhwc, (0, 3, 1, 2))
        out.append((short, label, nchw))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--no-xnnpack", action="store_true")
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(
            f"missing {args.model} — run: "
            "python3 benchmark/tools/export_host_onnx_assets.py --int8"
        )

    use_xnnpack = not args.no_xnnpack
    backend = backend_name(use_xnnpack)
    images = _load_images_nchw()
    num_images = len(images)

    session = make_session(
        args.model, num_threads=args.num_threads, use_xnnpack=use_xnnpack
    )
    in_name, in_shape, _in_dtype = input_meta(session)
    out_name = output_name(session)

    print("ONNX Runtime MobileNetV4 ImageNet int8 (QDQ) benchmark")
    print("  runtime:     ONNX Runtime (host)")
    print(f"  backend:     {backend}")
    print(f"  providers:   {session.get_providers()}")
    print(f"  model:       {args.model}")
    print(
        f"  method:      {num_images} images x {args.loops} loops = "
        f"{num_images * args.loops} invokes"
    )

    samples: list[float] = []
    correct = 0
    for loop in range(args.loops):
        for i, (short, label, batch) in enumerate(images):
            feeds = {in_name: batch.reshape(in_shape).astype(np.float32, copy=False)}
            t0 = time.perf_counter()
            outs = session.run([out_name], feeds)
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1e6)
            logits = np.asarray(outs[0]).reshape(-1).astype(np.float32)
            pred = int(logits.argmax())
            if loop == 0:
                ok = pred == label
                correct += int(ok)
                print(
                    f"  image {i} ImageNet {short} (class {label})".ljust(48)
                    + f" label={label:4d} pred={pred:4d} {'OK' if ok else 'MISS'}"
                )

    cold_us = samples[0]
    first_pass_mean = statistics.fmean(samples[:num_images])
    warm = samples[num_images:]
    warm_mean = statistics.fmean(warm)
    warm_median = statistics.median(warm)
    top1 = 100.0 * correct / num_images

    print()
    print(f"ONNX Runtime MobileNetV4 ImageNet int8 summary ({backend})")
    print(f"  top-1 accuracy:   {correct} / {num_images}  ({top1:.1f}%)")
    print(f"  warm mean:        {warm_mean:9.3f} us ({warm_mean / 1000.0:7.3f} ms)")
    print(
        "BENCHMARK_SUMMARY runtime=onnxruntime model=mobilenetv4_imagenet dtype=int8 "
        f"backend={backend} threads={args.num_threads} top1_correct={correct} "
        f"top1_total={num_images} top1_pct={top1:.1f} ten_image_mean_us={first_pass_mean:.3f} "
        f"warm_median_us={warm_median:.3f} warm_mean_us={warm_mean:.3f} cold_us={cold_us:.3f} "
        f"invokes={len(samples)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
