#!/usr/bin/env python3
"""Parse MCU UART logs for MNIST CNN int8 benchmarks and report dequantized confidence.

Firmware emits raw int8 softmax outputs only. This script dequantizes offline for
comparison between netkit and TFLM runs.

TFLite int8 softmax output spec (netkit and typical TFLM full-int8 graphs):
  scale = 1/256, zero_point = -128
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# TFLite int8 softmax output quantization (matches include/quant_output.hpp).
SOFTMAX_OUTPUT_SCALE = 1.0 / 256.0
SOFTMAX_OUTPUT_ZERO_POINT = -128


@dataclass(frozen=True)
class DigitResult:
    runtime: str
    image: int
    label: int
    pred: int
    pred_i8: int
    ok: int
    out_i8: tuple[int, ...]


DIGIT_SUMMARY_RE = re.compile(
    r"^DIGIT_SUMMARY\s+"
    r"runtime=(\w+)\s+"
    r"model=(\w+)\s+"
    r"image=(\d+)\s+"
    r"label=(\d+)\s+"
    r"pred=(\d+)\s+"
    r"pred_i8=(-?\d+)\s+"
    r"ok=(\d+)"
    r"(?:\s+out_i8=([-\d,]+))?"
)


def dequantize_softmax_i8(
    value: int,
    scale: float = SOFTMAX_OUTPUT_SCALE,
    zero_point: int = SOFTMAX_OUTPUT_ZERO_POINT,
) -> float:
    return (float(value) - float(zero_point)) * scale


def parse_digit_summaries(text: str) -> list[DigitResult]:
    results: list[DigitResult] = []
    for line in text.splitlines():
        m = DIGIT_SUMMARY_RE.search(line.strip())
        if not m:
            continue
        out_raw = m.group(8)
        out_i8: tuple[int, ...]
        if out_raw:
            out_i8 = tuple(int(x) for x in out_raw.split(","))
        else:
            out_i8 = ()
        results.append(
            DigitResult(
                runtime=m.group(1),
                image=int(m.group(3)),
                label=int(m.group(4)),
                pred=int(m.group(5)),
                pred_i8=int(m.group(6)),
                ok=int(m.group(7)),
                out_i8=out_i8,
            )
        )
    return results


def confidence_from_result(
    row: DigitResult,
    scale: float,
    zero_point: int,
) -> float:
    if row.out_i8 and 0 <= row.pred < len(row.out_i8):
        return dequantize_softmax_i8(row.out_i8[row.pred], scale, zero_point)
    return dequantize_softmax_i8(row.pred_i8, scale, zero_point)


def print_table(
    rows: list[DigitResult],
    scale: float,
    zero_point: int,
    title: str,
) -> None:
    if not rows:
        print(f"{title}: no DIGIT_SUMMARY lines found", file=sys.stderr)
        return

    rows = sorted(rows, key=lambda r: r.image)
    runtime = rows[0].runtime
    correct = sum(r.ok for r in rows)
    print(title)
    print(f"  runtime={runtime}  digits={len(rows)}  accuracy={correct}/{len(rows)}")
    print(f"  dequant: scale={scale:g} zero_point={zero_point}")
    print()
    print("  image  label  pred  pred_i8   conf       ok")
    for r in rows:
        conf = confidence_from_result(r, scale, zero_point)
        print(
            f"  {r.image:5d}  {r.label:5d}  {r.pred:4d}  {r.pred_i8:7d}  {conf:9.6f}  {'yes' if r.ok else 'no'}"
        )
    print()


def print_compare(
    left: list[DigitResult],
    right: list[DigitResult],
    scale: float,
    zero_point: int,
) -> None:
    left_by_image = {r.image: r for r in left}
    right_by_image = {r.image: r for r in right}
    images = sorted(set(left_by_image) | set(right_by_image))
    if not images:
        print("No DIGIT_SUMMARY lines to compare", file=sys.stderr)
        return

    left_rt = left[0].runtime if left else "?"
    right_rt = right[0].runtime if right else "?"
    print(f"netkit vs TFLM per-digit comparison (dequant conf, scale={scale:g}, zp={zero_point})")
    print()
    print(
        "  image  label  "
        f"{left_rt:>6} pred  {left_rt:>6} conf  "
        f"{right_rt:>6} pred  {right_rt:>6} conf  match"
    )
    for image in images:
        l = left_by_image.get(image)
        r = right_by_image.get(image)
        label = l.label if l else (r.label if r else -1)
        l_pred = l.pred if l else -1
        r_pred = r.pred if r else -1
        l_conf = confidence_from_result(l, scale, zero_point) if l else float("nan")
        r_conf = confidence_from_result(r, scale, zero_point) if r else float("nan")
        preds_match = l is not None and r is not None and l.pred == r.pred
        print(
            f"  {image:5d}  {label:5d}  "
            f"{l_pred:6d}  {l_conf:9.6f}  "
            f"{r_pred:6d}  {r_conf:9.6f}  "
            f"{'yes' if preds_match else 'no'}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse MCU CNN int8 UART logs and compute dequantized confidence offline."
    )
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        help="UART capture file (default: stdin)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("LOG_A", "LOG_B"),
        type=Path,
        help="Compare two logs side-by-side (e.g. netkit vs tflm)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=SOFTMAX_OUTPUT_SCALE,
        help=f"int8 softmax output scale (default: {SOFTMAX_OUTPUT_SCALE})",
    )
    parser.add_argument(
        "--zero-point",
        type=int,
        default=SOFTMAX_OUTPUT_ZERO_POINT,
        help=f"int8 softmax output zero point (default: {SOFTMAX_OUTPUT_ZERO_POINT})",
    )
    args = parser.parse_args()

    if args.compare:
        text_a = args.compare[0].read_text()
        text_b = args.compare[1].read_text()
        rows_a = parse_digit_summaries(text_a)
        rows_b = parse_digit_summaries(text_b)
        print_compare(rows_a, rows_b, args.scale, args.zero_point)
        return 0

    text = args.log.read_text() if args.log else sys.stdin.read()
    rows = parse_digit_summaries(text)
    runtime = rows[0].runtime if rows else "unknown"
    print_table(rows, args.scale, args.zero_point, title=f"MCU CNN int8 results ({runtime})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
