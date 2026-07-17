# Raspberry Pi Zero 2 W — MPU peer A/B

Arm **MPU** peer for netkit vs **TF Lite / LiteRT** on a Raspberry Pi Zero 2 W
(`aarch64`, Cortex-A53). This is the Linux/XNNPACK counterpart to the bare-metal
NUCLEO-F446RE MCU boards under `boards/nucleo-f446re-*`.

| Item | Value |
|------|--------|
| Board | Raspberry Pi Zero 2 W |
| ISA / OS | `linux/aarch64` (64-bit Raspberry Pi OS) |
| netkit target | `NETKIT_TARGET=mpu_arm` (XNNPACK on/off) |
| Peer | TF Lite / LiteRT Python benches (`benchmark/tflite/`) |
| Models | MNIST CNN, MNIST DS-CNN, MobileNetV4-Conv-Small ImageNet — **float32** and **int8** |
| Fairness | Same as host suite: 1 thread, discard 1st process, order swaps, `NETKIT_IM2COL=0` |

**Results (canonical):** [docs/STATUS.md](../../docs/STATUS.md#mpu--raspberry-pi-zero-2-w-aarch64-jul-2026) and the top of [README.md](../../README.md). Raw logs:

- [`benchmark/host_ab_suite_results_float32_pi_zero2w.txt`](../../benchmark/host_ab_suite_results_float32_pi_zero2w.txt)
- [`benchmark/host_ab_suite_results_int8_pi_zero2w.txt`](../../benchmark/host_ab_suite_results_int8_pi_zero2w.txt)

Unlike the NUCLEO firmware trees, this board has **no on-device Makefile firmware**.
netkit benches are **cross-built on the host** (Docker `linux/arm64`) and deployed
over SSH as a lean payload. Staging dirs under `benchmark/mpu_pi/` are local-only
(gitignored binaries / stage trees).

---

## Prerequisites

### Host (build machine)

- Docker with `linux/arm64` (used by `tools/build_mpu_pi_aarch64.sh`)
- XNNPACK sources fetched once: `./tools/fetch_xnnpack.sh` (or `make xnnpack-init`)
- Model / TF Lite fixtures already in-tree (or regenerated via the usual export
  targets under `benchmark/tflm/`)
- `expect` (for `tools/pi_ssh.sh` password SSH/SCP)

### Pi Zero 2 W

- 64-bit Raspberry Pi OS, SSH enabled, reachable on the LAN
- Enough free space for a venv + LiteRT wheel + lean payload (~hundreds of MiB)
- Default remote user/path assumptions below (override with env vars)

---

## Build (cross-compile on host)

From the **repo root**:

```bash
./tools/fetch_xnnpack.sh                 # once
./tools/build_mpu_pi_aarch64.sh          # float32 + int8, XNNPACK + reference ELFs
# or:
./tools/build_mpu_pi_aarch64.sh --dtype int8
./tools/build_mpu_pi_aarch64.sh --dtype float32 --skip-xnnpack
```

Outputs:

| Path | Contents |
|------|----------|
| `benchmark/mpu_pi/bin/*_pi_f32_{xnn,ref}` | float32 netkit benches |
| `benchmark/mpu_pi/bin/*_pi_i8_{xnn,ref}` | int8 netkit benches |
| `third_party/XNNPACK/netkit_lib_linux_aarch64/` | aarch64 XNNPACK static libs |

Optional: `NETKIT_PI_DOCKER_IMAGE=debian:trixie` (default) to override the Docker image.

---

## Run A/B on the Pi

Set connection env vars, then deploy + run:

```bash
export NETKIT_PI_HOST=192.168.0.176   # Pi IP (default in scripts)
export NETKIT_PI_USER=pi
export NETKIT_PI_PASS='your-password'
# optional: NETKIT_PI_DIR=/home/pi/netkit_mpu_ab

NETKIT_PI_PASS="$NETKIT_PI_PASS" ./tools/run_mpu_pi_float32_ab.sh
NETKIT_PI_PASS="$NETKIT_PI_PASS" ./tools/run_mpu_pi_int8_ab.sh
```

Each script:

1. Stages a **lean** tree under `benchmark/mpu_pi/stage` or `stage_int8`
   (benches, models, TF Lite Python peers, minimal `python/netkit` for fixtures)
2. Copies it to the Pi via `tools/pi_ssh.sh`
3. Installs a remote venv + LiteRT / deps
4. Runs order-averaged XNNPACK ON/OFF A/B (netkit ELF vs TF Lite Python)
5. Writes results back to `benchmark/host_ab_suite_results_{float32,int8}_pi_zero2w.txt`

TF Lite OFF uses `BUILTIN_REF` (same host-suite policy). See the host peer
footnote in [README.md](../../README.md).

---

## netkit profile on the Pi

| Setting | Value |
|---------|--------|
| Target | `mpu_arm` / `NETKIT_TARGET_MPU_ARM` |
| Accel ON | `NETKIT_XNNPACK=1` (LayerFast + qs8) |
| Accel OFF | reference / QuantOps (`*_pi_*_ref` ELFs) |
| CMSIS-NN | off (MCU-only) |
| Threads | 1 (matched to TF Lite) |
| `NETKIT_IM2COL` | `0` |

---

## Models covered

| Suite key | Float32 | Int8 |
|-----------|---------|------|
| MNIST CNN | `mnist_cnn` | `mnist_cnn_int8` |
| MNIST DS-CNN | `mnist_cnn_dw` | `mnist_cnn_dw_int8` |
| MobileNetV4-Small ImageNet | `mobilenetv4_imagenet_f32` | `mobilenetv4_imagenet_int8` |

---

## Related scripts

| Script | Role |
|--------|------|
| `tools/build_mpu_pi_aarch64.sh` | Docker cross-build of netkit + XNNPACK for aarch64 |
| `tools/run_mpu_pi_float32_ab.sh` | Stage, deploy, run float32 suite |
| `tools/run_mpu_pi_int8_ab.sh` | Stage, deploy, run int8 suite |
| `tools/pi_ssh.sh` | `expect`-based ssh/scp/rsync helper |

Host CPU three-way (Apple Silicon vs TF Lite + ORT) is separate — see
[docs/STATUS.md](../../docs/STATUS.md#host-three-way-suite-netkit-vs-tf-lite-vs-onnx-runtime)
and [benchmark/README.md](../../benchmark/README.md). MCU peers live under
`boards/nucleo-f446re-*`.
