#!/usr/bin/env python3
"""Deprecated entry point — use the dtype-named scripts instead.

  python3 benchmark/tools/run_host_ab_suite_float32.py
  python3 benchmark/tools/run_host_ab_suite_int8.py

This wrapper forwards to the float32 suite for compatibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from host_ab_suite_common import FLOAT32, cli_entry, ensure_assets_float32


def main() -> None:
    print(
        "NOTE: run_host_ab_suite.py is deprecated; "
        "prefer run_host_ab_suite_float32.py or run_host_ab_suite_int8.py\n"
        "Forwarding to FLOAT32 suite...\n",
        file=sys.stderr,
        flush=True,
    )
    cli_entry(
        FLOAT32,
        ensure_assets_float32,
        "Host A/B suite (float32; via deprecated wrapper)"
    )


if __name__ == "__main__":
    main()
