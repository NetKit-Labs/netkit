#!/usr/bin/env python3
"""Render MNIST benchmark comparison tables as PNG images."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.table import Table


def speedup(nk: float | None, tf: float) -> str:
    if nk is None or nk <= 0 or tf <= 0:
        return "—"
    ratio = tf / nk
    if abs(ratio - 1.0) < 0.05:
        return "tie"
    if ratio > 1.0:
        return f"{ratio:.1f}x faster"
    return f"{1.0 / ratio:.1f}x slower"


def fmt_us(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}"


def style_table(table: Table, header_rows: int = 1) -> None:
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.8)
        if row < header_rows:
            cell.set_facecolor("#2d3748")
            cell.set_text_props(color="white", weight="bold", fontsize=11)
        elif row % 2 == 0:
            cell.set_facecolor("#f7fafc")
        else:
            cell.set_facecolor("#ffffff")
        cell.set_height(0.08)
        cell.PAD = 0.04


def parse_benchmark_summaries(text: str) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    pat = re.compile(
        r"BENCHMARK_SUMMARY runtime=(\w+) model=(\w+) backend=([\w-]+) mean_us=([0-9.+-eE]+)"
    )
    for m in pat.finditer(text):
        out[(m.group(1), m.group(2), m.group(3))] = float(m.group(4))
    return out


def normalize_tag(tag: str) -> str:
    aliases = {
        "conv2d": "Conv2D",
        "CONV_2D": "Conv2D",
        "max_pool2d": "MaxPool2D",
        "MAX_POOL_2D": "MaxPool2D",
        "reshape": "Reshape",
        "RESHAPE": "Reshape",
        "flatten": "Reshape",
        "fully_connected": "FullyConnected",
        "FULLY_CONNECTED": "FullyConnected",
        "dense": "FullyConnected",
        "softmax": "Softmax",
        "SOFTMAX": "Softmax",
    }
    return aliases.get(tag, aliases.get(tag.lower(), tag))


def parse_profile_summaries(
    text: str, model: str
) -> tuple[dict[str, float], dict[str, tuple[float | None, float | None]]]:
    meta_nk: dict[str, float] = {}
    meta_tf: dict[str, float] = {}
    ops: dict[str, tuple[float | None, float | None]] = {}

    pat = re.compile(
        rf"PROFILE_SUMMARY runtime=(\w+) model={model} kind=(\w+) tag=(\S+) mean_us=([0-9.+-eE]+)"
    )
    for m in pat.finditer(text):
        runtime, kind, tag, mean_s = m.group(1), m.group(2), m.group(3), m.group(4)
        mean = float(mean_s)
        if kind == "meta":
            if runtime == "netkit":
                meta_nk[tag] = mean
            elif runtime == "tflm":
                meta_tf[tag] = mean
        elif kind == "op":
            key = normalize_tag(tag)
            nk_val, tf_val = ops.get(key, (None, None))
            if runtime == "netkit":
                nk_val = (nk_val or 0.0) + mean
            elif runtime == "tflm":
                tf_val = (tf_val or 0.0) + mean
            ops[key] = (nk_val, tf_val)

    meta = meta_nk
    meta["_tf_wall"] = meta_tf.get("wall_clock", 0.0)
    return meta, ops


def render_latency_table(rows: list[tuple[str, str, float, float]], out_path: Path) -> None:
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(11.5, 4.2), dpi=180)
    ax.axis("off")

    fig.suptitle(
        "MNIST mean invoke — NETKIT vs TFLM",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    ax.text(
        0.5,
        0.92,
        "100 runs x 10 images · first invoke discarded each run · mean only\n"
        "Speedup = TFLM us / NETKIT us",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=10,
        color="#4a5568",
    )

    col_labels = ["Model", "Variant", "NETKIT (us)", "TFLM (us)", "Speedup"]
    cell_text = [
        [model, variant, fmt_us(nk), fmt_us(tf), speedup(nk, tf)]
        for model, variant, nk, tf in rows
    ]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        bbox=[0.02, 0.08, 0.96, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.35)

    col_widths = [0.10, 0.36, 0.14, 0.14, 0.18]
    for col, width in enumerate(col_widths):
        for row in range(len(cell_text) + 1):
            table[(row, col)].set_width(width)

    style_table(table)

    for col in range(len(col_labels)):
        table[(2, col)].set_edgecolor("#718096")
        table[(2, col)].set_linewidth(2.0)

    fig.savefig(out_path, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close(fig)


def render_profile_table(
    model: str,
    profile_rows: list[tuple[str, float | None, float | None]],
    wall_nk: float | None,
    wall_tf: float | None,
    out_path: Path,
) -> None:
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=180)
    ax.axis("off")

    subtitle = (
        "100 runs x 10 images · per-layer / per-op mean us per invoke\n"
        "NETKIT last Dense includes fused Softmax; TFLM times Softmax separately"
    )
    if wall_nk and wall_tf and wall_nk > 0:
        wall_speedup = wall_tf / wall_nk
        subtitle = (
            f"100 runs x 10 images · per-layer / per-op mean us per invoke\n"
            f"End-to-end: TFLM {wall_tf:.1f} us vs NETKIT {wall_nk:.1f} us "
            f"({wall_speedup:.1f}x faster on NETKIT)\n"
            "NETKIT last Dense includes fused Softmax; TFLM times Softmax separately"
        )

    fig.suptitle(
        f"{model.upper()} per-op profile — NETKIT vs TFLM",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    ax.text(
        0.5,
        0.90,
        subtitle,
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#4a5568",
    )

    col_labels = ["Op", "NETKIT (us)", "TFLM (us)", "Speedup"]
    cell_text = []
    for op, nk, tf in profile_rows:
        if nk is None and tf is None:
            continue
        if nk is not None and tf is not None:
            sp = speedup(nk, tf)
        elif nk is not None:
            sp = "fused in Dense"
        else:
            sp = "—"
        cell_text.append([op, fmt_us(nk), fmt_us(tf), sp])

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        bbox=[0.06, 0.12, 0.88, 0.58],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.4)

    col_widths = [0.28, 0.18, 0.18, 0.26]
    for col, width in enumerate(col_widths):
        for row in range(len(cell_text) + 1):
            table[(row, col)].set_width(width)

    style_table(table)

    fig.savefig(out_path, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close(fig)


def build_latency_rows(summaries: dict[tuple[str, str, str], float]) -> list[tuple[str, str, float, float]]:
    tf_mlp = summaries.get(("tflm", "mlp", "reference"))
    tf_cnn = summaries.get(("tflm", "cnn", "reference"))
    nk_dsp_mlp = summaries.get(("netkit", "mlp", "cmsis-dsp"))
    nk_ref_mlp = summaries.get(("netkit", "mlp", "reference"))
    nk_dsp_cnn = summaries.get(("netkit", "cnn", "cmsis-dsp"))
    nk_ref_cnn = summaries.get(("netkit", "cnn", "reference"))

    rows: list[tuple[str, str, float, float]] = []
    if nk_dsp_mlp is not None and tf_mlp is not None:
        rows.append(("MLP", "NETKIT (with CMSIS-DSP)", nk_dsp_mlp, tf_mlp))
    if nk_ref_mlp is not None and tf_mlp is not None:
        rows.append(("MLP", "NETKIT (without CMSIS-DSP)", nk_ref_mlp, tf_mlp))
    if nk_dsp_cnn is not None and tf_cnn is not None:
        rows.append(("CNN", "NETKIT (with CMSIS-DSP)", nk_dsp_cnn, tf_cnn))
    if nk_ref_cnn is not None and tf_cnn is not None:
        rows.append(("CNN", "NETKIT (without CMSIS-DSP)", nk_ref_cnn, tf_cnn))
    return rows


def build_profile_rows(
    merged_ops: dict[str, tuple[float | None, float | None]],
) -> list[tuple[str, float | None, float | None]]:
    def sort_key(item: tuple[str, tuple[float | None, float | None]]) -> float:
        nk, tf = item[1]
        return max(nk or 0.0, tf or 0.0)

    return [
        (tag, nk, tf)
        for tag, (nk, tf) in sorted(merged_ops.items(), key=sort_key, reverse=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="compare.sh capture log (BENCHMARK_SUMMARY / PROFILE_SUMMARY lines)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory for PNG output (default: benchmark/)",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    text = args.log.read_text(encoding="utf-8", errors="replace")
    summaries = parse_benchmark_summaries(text)
    latency_rows = build_latency_rows(summaries)

    mlp_meta, mlp_ops = parse_profile_summaries(text, "mlp")
    cnn_meta, cnn_ops = parse_profile_summaries(text, "cnn")

    latency_path = args.out_dir / "mnist_latency_comparison.png"
    mlp_profile_path = args.out_dir / "mnist_mlp_profile_comparison.png"
    cnn_profile_path = args.out_dir / "mnist_cnn_profile_comparison.png"

    if latency_rows:
        render_latency_table(latency_rows, latency_path)
        print(f"Wrote {latency_path}")

    if mlp_ops:
        tf_wall = mlp_meta.get("_tf_wall") or None
        if tf_wall == 0.0:
            tf_wall = None
        render_profile_table(
            "MLP",
            build_profile_rows(mlp_ops),
            mlp_meta.get("wall_clock"),
            tf_wall,
            mlp_profile_path,
        )
        print(f"Wrote {mlp_profile_path}")

    if cnn_ops:
        tf_wall = cnn_meta.get("_tf_wall") or None
        if tf_wall == 0.0:
            tf_wall = None
        render_profile_table(
            "CNN",
            build_profile_rows(cnn_ops),
            cnn_meta.get("wall_clock"),
            tf_wall,
            cnn_profile_path,
        )
        print(f"Wrote {cnn_profile_path}")


if __name__ == "__main__":
    main()
