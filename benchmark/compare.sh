#!/usr/bin/env bash
# Run netkit + TFLM MNIST benchmarks and print comparison tables.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GMAKE="${GMAKE:-$(command -v gmake 2>/dev/null || command -v gmak 2>/dev/null || true)}"
if [[ -z "$GMAKE" && -x /opt/homebrew/bin/gmake ]]; then
  GMAKE=/opt/homebrew/bin/gmake
fi

echo "Building and running benchmarks..."
echo

run_netkit() {
  make -C benchmark/netkit "$1" 2>&1
}

run_tflm() {
  if [[ -n "$GMAKE" ]]; then
    GMAKE="$GMAKE" make -C benchmark/tflm "$1" 2>&1
  else
    echo "GNU make (gmake) not found; skipping TFLM $1" >&2
    return 0
  fi
}

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

{
  run_netkit run-mlp
  echo "---"
  run_netkit run-cnn
  echo "---"
  run_tflm run-mlp
  echo "---"
  run_tflm run-cnn
  echo "---"
  run_netkit run-mlp-profile
  echo "---"
  run_netkit run-cnn-profile
  echo "---"
  run_tflm run-mlp-profile
  echo "---"
  run_tflm run-cnn-profile
} | tee "$TMP"

parse_summary() {
  local runtime="$1"
  local model="$2"
  local backend="$3"
  grep "BENCHMARK_SUMMARY runtime=${runtime} model=${model} backend=${backend} " "$TMP" | tail -1 | \
    sed -n 's/.*mean_us=\([^ ]*\) runs=.*/\1/p'
}

load_result() {
  local var_prefix="$1"
  local runtime="$2"
  local model="$3"
  local backend="$4"
  local mean
  if read -r mean <<< "$(parse_summary "$runtime" "$model" "$backend" || echo 'nan')"; then
    if [[ "$mean" != "nan" && -n "$mean" ]]; then
      eval "${var_prefix}_mean=$mean"
      return 0
    fi
  fi
  eval "${var_prefix}_mean=nan"
  return 1
}

load_result nk_ref_mlp netkit mlp reference || true
load_result tf_ref_mlp tflm mlp reference || true

load_result nk_ref_cnn netkit cnn reference || true
load_result tf_ref_cnn tflm cnn reference || true

print_table() {
  python3 - \
    "$nk_ref_mlp_mean" "$tf_ref_mlp_mean" \
    "$nk_ref_cnn_mean" "$tf_ref_cnn_mean" <<'PY'
import sys

def fval(s):
    try:
        v = float(s)
        return v if v > 0 else None
    except ValueError:
        return None

def fmt_us(v):
    return f"{v:.1f}"

def fmt_delta(nk, tf):
    if nk is None or tf is None or tf <= 0 or nk <= 0:
        return "-"
    speedup = tf / nk
    if abs(speedup - 1.0) < 0.0005:
        return "tie"
    if speedup > 1.0:
        return f"{speedup:.1f}× faster"
    return f"{1.0 / speedup:.1f}× slower"

args = sys.argv[1:]
(
    nk_ref_mlp, tf_mlp,
    nk_ref_cnn, tf_cnn,
) = [fval(a) for a in args]

nk_variants = [
    ("NETKIT (reference)", nk_ref_mlp, nk_ref_cnn),
]

def print_model_table(title, tf_baseline, nk_idx):
    print()
    print(title)
    print("-" * len(title))
    if tf_baseline is not None:
        print(f"TFLM reference baseline: {fmt_us(tf_baseline)} µs mean invoke")
    print()
    print(
        f"{'Variant':<28} | {'NETKIT µs':>11} | {'TFLM µs':>10} | Speedup"
    )
    print(f"{'-'*28}-+-{'-'*11}-+-{'-'*10}-+-{'-'*14}")
    for label, mlp, cnn in nk_variants:
        nk = (mlp, cnn)[nk_idx]
        if nk is None and tf_baseline is None:
            continue
        nk_str = fmt_us(nk) if nk is not None else "-"
        tf_str = fmt_us(tf_baseline) if tf_baseline is not None else "-"
        print(f"{label:<28} | {nk_str:>11} | {tf_str:>10} | {fmt_delta(nk, tf_baseline)}")

print()
print("=" * 82)
print("MNIST mean invoke — NETKIT vs TFLM (side by side)")
print("=" * 82)
print("Speedup = TFLM mean invoke ÷ NETKIT mean invoke (e.g. 6× = NETKIT is six times faster).")
print_model_table("MLP", tf_mlp, 0)
print_model_table("CNN", tf_cnn, 1)
print()
PY
}

print_profile_table() {
  local model="$1"
  local title="$2"
  local note="$3"
  python3 - "$TMP" "$model" "$title" "$note" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
model = sys.argv[2]
title = sys.argv[3]
note = sys.argv[4]

def parse_profiles(runtime):
    meta = {}
    ops = {}
    for line in text.splitlines():
        if f"PROFILE_SUMMARY runtime={runtime} model={model}" not in line:
            continue
        kind = re.search(r"kind=([^ ]+)", line)
        tag = re.search(r"tag=([^ ]+)", line)
        mean = re.search(r"mean_us=([0-9.+-eE]+)", line)
        pct = re.search(r"pct=([0-9.+-eE]+)", line)
        if not kind or not tag or not mean:
            continue
        entry = {
            "kind": kind.group(1),
            "tag": tag.group(1),
            "mean_us": float(mean.group(1)),
            "pct": float(pct.group(1)) if pct else None,
        }
        if entry["kind"] == "meta":
            meta[entry["tag"]] = entry
        elif entry["kind"] == "op":
            ops[entry["tag"]] = entry
    return meta, ops

def normalize_tag(tag):
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

nk_meta, nk_ops = parse_profiles("netkit")
tf_meta, tf_ops = parse_profiles("tflm")

if not nk_ops and not tf_ops:
    print(f"({model.upper()} profile data not found — run-{model}-profile may have been skipped)")
    sys.exit(0)

def merge_ops(src):
    merged = {}
    for tag, row in src.items():
        key = normalize_tag(tag)
        if key in merged:
            merged[key]["mean_us"] += row["mean_us"]
        else:
            merged[key] = dict(row)
            merged[key]["tag"] = key
    return merged

nk_ops = merge_ops(nk_ops)
tf_ops = merge_ops(tf_ops)

print()
print("=" * 92)
print(title)
print("=" * 92)
print("100 runs × 10 images; first invoke discarded each run. Op times from in-forward profilers.")
print(note)
print()

def print_meta_block(label, meta):
    if not meta:
        return
    wall = meta.get("wall_clock", {}).get("mean_us")
    print(f"{label} summary")
    print("-" * (len(label) + 8))
    print(f"{'Metric':<24} | {'mean (us)':>10} | {'% wall':>8}")
    print(f"{'-'*24}-+-{'-'*10}-+-{'-'*8}")
    order = ["wall_clock", "op_time_sum", "between_op_overhead"]
    labels = {
        "wall_clock": "Invoke wall clock",
        "op_time_sum": "Sum of op times",
        "between_op_overhead": "Between-op gap",
    }
    for key in order:
        item = meta.get(key)
        if not item:
            continue
        pct = item.get("pct")
        if pct is None and wall and wall > 0:
            pct = (item["mean_us"] / wall) * 100.0
        pct_str = f"{pct:.1f}" if pct is not None else "-"
        print(f"{labels.get(key, key):<24} | {item['mean_us']:10.1f} | {pct_str:>8}")
    print()

nk_wall = nk_meta.get("wall_clock", {}).get("mean_us")
tf_wall = tf_meta.get("wall_clock", {}).get("mean_us")

print_meta_block("NETKIT", nk_meta)
print_meta_block("TFLM", tf_meta)

if nk_wall and tf_wall and nk_wall > 0:
    wall_speedup = tf_wall / nk_wall
    print(f"End-to-end invoke: TFLM {tf_wall:.1f} µs vs NETKIT {nk_wall:.1f} µs → {wall_speedup:.1f}× faster on NETKIT")
    print()

all_tags = sorted(set(nk_ops) | set(tf_ops), key=lambda t: (
    -(nk_ops.get(t, tf_ops.get(t, {})).get("mean_us", 0)),
    t,
))

print("Per-op comparison (sorted by max mean time)")
print("--------------------------------")
print(
    f"{'Op':<16} | {'NETKIT µs':>10} | {'TFLM µs':>10} | Speedup"
)
print(f"{'-'*16}-+-{'-'*10}-+-{'-'*10}-+-{'-'*14}")

def fmt_delta(nk, tf):
    if nk is None or tf is None or tf <= 0 or nk <= 0:
        return "-"
    speedup = tf / nk
    if abs(speedup - 1.0) < 0.0005:
        return "tie"
    if speedup > 1.0:
        return f"{speedup:.1f}× faster"
    return f"{1.0 / speedup:.1f}× slower"

for tag in all_tags:
    nk = nk_ops.get(tag)
    tf = tf_ops.get(tag)
    nk_us = nk["mean_us"] if nk else None
    tf_us = tf["mean_us"] if tf else None
    nk_us_s = f"{nk_us:.1f}" if nk_us is not None else "-"
    tf_us_s = f"{tf_us:.1f}" if tf_us is not None else "-"
    print(
        f"{tag:<16} | {nk_us_s:>10} | {tf_us_s:>10} | {fmt_delta(nk_us, tf_us)}"
    )

print()
PY
}

echo
echo "================================================================="
echo "MNIST invoke comparison (100 runs, first invoke discarded per run)"
echo "================================================================="

print_table
echo "================================================================="
print_profile_table "mlp" \
  "MLP per-op profile — NETKIT vs TFLM (mean µs per invoke)" \
  "NETKIT last Dense includes fused Softmax; TFLM times Softmax separately."
print_profile_table "cnn" \
  "CNN per-op profile — NETKIT vs TFLM (mean µs per invoke)" \
  "NETKIT last Dense includes fused Softmax; TFLM times Softmax separately."
echo "================================================================="

# Render PNG tables from captured log.
python3 benchmark/tools/render_benchmark_tables.py --log "$TMP" --out-dir benchmark
