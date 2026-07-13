#!/usr/bin/env bash
# Deploy lean int8 A/B payload to Pi Zero 2 W and run the suite.
# Prerequisites: ./tools/build_mpu_pi_aarch64.sh --dtype int8
#
#   NETKIT_PI_PASS='...' ./tools/run_mpu_pi_int8_ab.sh
#
# Float32 results file is left untouched.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${NETKIT_PI_HOST:-192.168.0.176}"
USER="${NETKIT_PI_USER:-pi}"
PASS="${NETKIT_PI_PASS:?set NETKIT_PI_PASS}"
REMOTE="${NETKIT_PI_DIR:-/home/pi/netkit_mpu_ab}"
STAGE="$ROOT/benchmark/mpu_pi/stage_int8"
OUT="$ROOT/benchmark/host_ab_suite_results_int8_pi_zero2w.txt"
SSH="$ROOT/tools/pi_ssh.sh"
export NETKIT_PI_HOST="$HOST" NETKIT_PI_USER="$USER" NETKIT_PI_PASS="$PASS"

need() { [[ -f "$1" ]] || { echo "missing $1 — build first (./tools/build_mpu_pi_aarch64.sh --dtype int8)" >&2; exit 1; }; }

for m in cnn cnn_dw imagenet; do
  case "$m" in
    cnn) p=mnist_cnn_int8_bench ;;
    cnn_dw) p=mnist_cnn_dw_int8_bench ;;
    imagenet) p=mobilenetv4_imagenet_int8_bench ;;
  esac
  need "$ROOT/benchmark/mpu_pi/bin/${p}_pi_i8_ref"
  need "$ROOT/benchmark/mpu_pi/bin/${p}_pi_i8_xnn"
done
need "$ROOT/models/mnist_cnn.nk"  # float TCAS source for TF Lite int8 image quant
need "$ROOT/benchmark/tflm/generated/mnist_cnn_int8_tflite_quant.json"
need "$ROOT/benchmark/tflm/generated/mnist_cnn_dw_int8_tflite_quant.json"

echo "Staging lean int8 tree under $STAGE ..."
rm -rf "$STAGE"
mkdir -p \
  "$STAGE/models" \
  "$STAGE/benchmark/mpu_pi/bin" \
  "$STAGE/benchmark/tflite" \
  "$STAGE/benchmark/tflm/generated" \
  "$STAGE/benchmark/tflm/tools" \
  "$STAGE/python"

cp -f "$ROOT/benchmark/mpu_pi/bin/"*_pi_i8_* "$STAGE/benchmark/mpu_pi/bin/"
cp -f \
  "$ROOT/models/mnist_cnn_int8.nk" \
  "$ROOT/models/mnist_cnn_dw_int8.nk" \
  "$ROOT/models/mobilenetv4_imagenet_int8.nk" \
  "$ROOT/models/mnist_cnn.nk" \
  "$STAGE/models/"
cp -f \
  "$ROOT/benchmark/tflm/generated/mnist_cnn_int8.tflite" \
  "$ROOT/benchmark/tflm/generated/mnist_cnn_dw_int8.tflite" \
  "$ROOT/benchmark/tflm/generated/mobilenetv4_imagenet_int8.tflite" \
  "$ROOT/benchmark/tflm/generated/mnist_cnn_int8_tflite_quant.json" \
  "$ROOT/benchmark/tflm/generated/mnist_cnn_dw_int8_tflite_quant.json" \
  "$STAGE/benchmark/tflm/generated/"
cp -a "$ROOT/benchmark/tflm/generated/imagenet_sample_cache" \
  "$STAGE/benchmark/tflm/generated/"
cp -f \
  "$ROOT/benchmark/tflite/mnist_cnn_int8_bench.py" \
  "$ROOT/benchmark/tflite/mobilenetv4_imagenet_int8_bench.py" \
  "$STAGE/benchmark/tflite/"
cp -f \
  "$ROOT/benchmark/tflm/tools/export_int8_test_images.py" \
  "$ROOT/benchmark/tflm/tools/export_imagenet_mnv4_test_images.py" \
  "$STAGE/benchmark/tflm/tools/"
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' \
  "$ROOT/python/netkit/" "$STAGE/python/netkit/"

cat > "$STAGE/run_ab_int8.sh" <<'REMOTE'
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

echo '######## MPU A/B SUITE — INT8 (Raspberry Pi Zero 2 W) ########'
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
        NK="$ROOT/benchmark/mpu_pi/bin/mnist_cnn_int8_bench_pi_i8_$mode"
        NK_MODEL="$ROOT/models/mnist_cnn_int8.nk"
        TF=("$PY" "$ROOT/benchmark/tflite/mnist_cnn_int8_bench.py" --num-threads 1 --runs 10
            --model "$ROOT/benchmark/tflm/generated/mnist_cnn_int8.tflite"
            --nk "$ROOT/models/mnist_cnn.nk"
            --quant-json "$ROOT/benchmark/tflm/generated/mnist_cnn_int8_tflite_quant.json")
        ;;
      cnn_dw)
        NK="$ROOT/benchmark/mpu_pi/bin/mnist_cnn_dw_int8_bench_pi_i8_$mode"
        NK_MODEL="$ROOT/models/mnist_cnn_dw_int8.nk"
        TF=("$PY" "$ROOT/benchmark/tflite/mnist_cnn_int8_bench.py" --num-threads 1 --runs 10
            --model "$ROOT/benchmark/tflm/generated/mnist_cnn_dw_int8.tflite"
            --nk "$ROOT/models/mnist_cnn.nk"
            --quant-json "$ROOT/benchmark/tflm/generated/mnist_cnn_dw_int8_tflite_quant.json")
        ;;
      imagenet)
        NK="$ROOT/benchmark/mpu_pi/bin/mobilenetv4_imagenet_int8_bench_pi_i8_$mode"
        NK_MODEL="$ROOT/models/mobilenetv4_imagenet_int8.nk"
        TF=("$PY" "$ROOT/benchmark/tflite/mobilenetv4_imagenet_int8_bench.py" --num-threads 1
            --model "$ROOT/benchmark/tflm/generated/mobilenetv4_imagenet_int8.tflite")
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
chmod +x "$STAGE/run_ab_int8.sh" "$STAGE/install_deps.sh" "$STAGE/cleanup_pass.sh" "$STAGE/benchmark/mpu_pi/bin/"*

echo "Ensuring remote dir + rsync ..."
"$SSH" ssh "mkdir -p $(printf %q "$REMOTE")"
"$SSH" rsync_to "$STAGE" "$REMOTE"

echo "Installing python deps on Pi if needed ..."
pass_tmp="$(mktemp)"
printf '%s\n' "$PASS" > "$pass_tmp"
chmod 600 "$pass_tmp"
"$SSH" scp "$pass_tmp" "$REMOTE/.netkit_pi_pass"
rm -f "$pass_tmp"
"$SSH" ssh "$REMOTE/install_deps.sh"
"$SSH" ssh "$REMOTE/cleanup_pass.sh"

echo "Running remote int8 A/B suite ..."
t0=$(date +%s)
"$SSH" ssh "$REMOTE/run_ab_int8.sh" | tee "$OUT"
t1=$(date +%s)
echo "# wall_s=$((t1 - t0))" >> "$OUT"
echo "Wrote $OUT (float32 results untouched)"
