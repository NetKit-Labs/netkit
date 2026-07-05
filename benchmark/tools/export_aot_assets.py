#!/usr/bin/env python3
"""Generate optimized AOT embed sources for MNIST benchmark models."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "netkit" / "generated" / "aot"

MODELS = [
    ROOT / "models" / "mnist_mlp.nk",
    ROOT / "models" / "mnist_cnn.nk",
]


def main() -> int:
    sys.path.insert(0, str(ROOT / "python"))
    from netkit.aot_compile import compile_aot

    missing = [p for p in MODELS if not p.is_file()]
    if missing:
        for path in missing:
            print(f"missing model: {path}", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    for nk_path in MODELS:
        result = compile_aot(nk_path, OUT, optimize=True)
        print(
            f"wrote AOT {result.network} {result.model_name} "
            f"(lowered={result.lowered}, arena={result.arena_bytes_recommended})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
