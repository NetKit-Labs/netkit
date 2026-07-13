#!/usr/bin/env bash
# Deploy lean float32 A/B payload to Pi Zero 2 W and run the suite.
# Prerequisites: tools/build_mpu_pi_aarch64.sh has produced benchmark/mpu_pi/bin/*
#
#   NETKIT_PI_PASS='...' ./tools/run_mpu_pi_float32_ab.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${NETKIT_PI_HOST:-192.168.0.176}"
USER="${NETKIT_PI_USER:-pi}"
PASS="${NETKIT_PI_PASS:?set NETKIT_PI_PASS}"
REMOTE="${NETKIT_PI_DIR:-/home/pi/netkit_mpu_ab}"
STAGE="$ROOT/benchmark/mpu_pi/stage"
OUT="$ROOT/benchmark/host_ab_suite_results_float32_pi_zero2w.txt"
SSH="$ROOT/tools/pi_ssh.sh"
export NETKIT_PI_HOST="$HOST" NETKIT_PI_USER="$USER" NETKIT_PI_PASS="$PASS"

need() { [[ -f "$1" ]] || { echo "missing $1 — build first" >&2; exit 1; }; }

for m in cnn cnn_dw imagenet; do
  case "$m" in
    cnn) p=mnist_cnn_bench ;;
    cnn_dw) p=mnist_cnn_dw_bench ;;
    imagenet) p=mobilenetv4_imagenet_bench ;;
  esac
  need "$ROOT/benchmark/mpu_pi/bin/${p}_pi_f32_ref"
  need "$ROOT/benchmark/mpu_pi/bin/${p}_pi_f32_xnn"
done

echo "Staging lean tree under $STAGE ..."
rm -rf "$STAGE"
mkdir -p \
  "$STAGE/models" \
  "$STAGE/benchmark/mpu_pi/bin" \
  "$STAGE/benchmark/tflite" \
  "$STAGE/benchmark/tflm/generated" \
  "$STAGE/benchmark/tflm/tools"

cp -f "$ROOT/benchmark/mpu_pi/bin/"*_pi_f32_* "$STAGE/benchmark/mpu_pi/bin/"
cp -f \
  "$ROOT/models/mnist_cnn.nk" \
  "$ROOT/models/mnist_cnn_dw.nk" \
  "$ROOT/models/mobilenetv4_imagenet_f32.nk" \
  "$STAGE/models/"
cp -f \
  "$ROOT/benchmark/tflm/generated/mnist_cnn.tflite" \
  "$ROOT/benchmark/tflm/generated/mnist_cnn_dw.tflite" \
  "$ROOT/benchmark/tflm/generated/mobilenetv4_imagenet_f32.tflite" \
  "$STAGE/benchmark/tflm/generated/"
cp -a "$ROOT/benchmark/tflm/generated/imagenet_sample_cache" \
  "$STAGE/benchmark/tflm/generated/"
cp -f \
  "$ROOT/benchmark/tflite/mnist_cnn_bench.py" \
  "$ROOT/benchmark/tflite/mobilenetv4_imagenet_bench.py" \
  "$STAGE/benchmark/tflite/"
# Imports used by the TF Lite benches.
cp -f \
  "$ROOT/benchmark/tflm/tools/export_int8_test_images.py" \
  "$ROOT/benchmark/tflm/tools/export_imagenet_mnv4_test_images.py" \
  "$STAGE/benchmark/tflm/tools/"

# TF Lite MNIST benches import netkit.reader for float TCAS images.
mkdir -p "$STAGE/python"
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' \
  "$ROOT/python/netkit/" "$STAGE/python/netkit/"

# Remote driver (lives at repo root on the Pi tree).
cat > "$STAGE/run_ab_float32.sh" <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

run_keep() {
  "$@" >/dev/null 2>&1 || true
  "$@"
}
summary() { grep -E '^BENCHMARK_SUMMARY' | tail -1; }

echo '######## MPU A/B SUITE — FLOAT32 (Raspberry Pi Zero 2 W) ########'
echo "host=$(hostname) arch=$(uname -m) $(grep -m1 Model /proc/cpuinfo | sed 's/\t/ /g')"
free -h | sed -n '1,2p'
echo

for mode in xnn ref; do
  if [[ "$mode" == xnn ]]; then
    echo '======== MODE: XNNPACK ON ========'
    TF_EXTRA=()
  else
    echo '======== MODE: XNNPACK OFF (reference) ========'
    TF_EXTRA=(--no-xnnpack)
  fi
  for model in cnn cnn_dw imagenet; do
    case "$model" in
      cnn)
        NK="$ROOT/benchmark/mpu_pi/bin/mnist_cnn_bench_pi_f32_$mode"
        NK_MODEL="$ROOT/models/mnist_cnn.nk"
        TF=("$PY" "$ROOT/benchmark/tflite/mnist_cnn_bench.py" --num-threads 1 --runs 10
            --model "$ROOT/benchmark/tflm/generated/mnist_cnn.tflite"
            --nk "$ROOT/models/mnist_cnn.nk")
        ;;
      cnn_dw)
        NK="$ROOT/benchmark/mpu_pi/bin/mnist_cnn_dw_bench_pi_f32_$mode"
        NK_MODEL="$ROOT/models/mnist_cnn_dw.nk"
        TF=("$PY" "$ROOT/benchmark/tflite/mnist_cnn_bench.py" --num-threads 1 --runs 10
            --model "$ROOT/benchmark/tflm/generated/mnist_cnn_dw.tflite"
            --nk "$ROOT/models/mnist_cnn_dw.nk")
        ;;
      imagenet)
        NK="$ROOT/benchmark/mpu_pi/bin/mobilenetv4_imagenet_bench_pi_f32_$mode"
        NK_MODEL="$ROOT/models/mobilenetv4_imagenet_f32.nk"
        TF=("$PY" "$ROOT/benchmark/tflite/mobilenetv4_imagenet_bench.py" --num-threads 1
            --model "$ROOT/benchmark/tflm/generated/mobilenetv4_imagenet_f32.tflite")
        ;;
    esac
    TF+=("${TF_EXTRA[@]}")
    echo "---- model=$model mode=$mode order=nk_then_tf ----"
    echo -n "netkit: "; run_keep "$NK" "$NK_MODEL" | summary
    echo -n "tflite: "; run_keep "${TF[@]}" | summary
    echo "---- model=$model mode=$mode order=tf_then_nk ----"
    echo -n "tflite: "; run_keep "${TF[@]}" | summary
    echo -n "netkit: "; run_keep "$NK" "$NK_MODEL" | summary
    echo
  done
done
echo DONE
REMOTE
cp -f "$ROOT/tools/pi_install_float32_deps.sh" "$STAGE/install_deps.sh"
cat > "$STAGE/cleanup_pass.sh" <<'CLN'
#!/bin/sh
rm -f "$(cd "$(dirname "$0")" && pwd)/.netkit_pi_pass"
CLN
chmod +x "$STAGE/cleanup_pass.sh"
chmod +x "$STAGE/run_ab_float32.sh" "$STAGE/install_deps.sh" "$STAGE/benchmark/mpu_pi/bin/"*

echo "Ensuring remote dir + rsync ..."
"$SSH" ssh "mkdir -p $(printf %q "$REMOTE")"
"$SSH" rsync_to "$STAGE" "$REMOTE"

echo "Installing python deps on Pi (venv + litert/tflite fallbacks) ..."
# Passfile via scp (rsync excludes .netkit_pi_pass / .venv).
pass_tmp="$(mktemp)"
printf '%s
' "$PASS" > "$pass_tmp"
chmod 600 "$pass_tmp"
"$SSH" scp "$pass_tmp" "$REMOTE/.netkit_pi_pass"
rm -f "$pass_tmp"
"$SSH" ssh "$REMOTE/install_deps.sh"
"$SSH" ssh "$REMOTE/cleanup_pass.sh"

echo "Running remote float32 A/B suite ..."
t0=$(date +%s)
"$SSH" ssh "$REMOTE/run_ab_float32.sh" \
  | tee "$OUT"
t1=$(date +%s)
echo "# wall_s=$((t1 - t0))" >> "$OUT"
echo "Wrote $OUT"
