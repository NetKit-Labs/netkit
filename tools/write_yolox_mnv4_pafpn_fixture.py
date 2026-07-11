#!/usr/bin/env python3
"""Alias: regenerate models/yolox_mnv4_small.nk (PAFPN multi-scale detector)."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("write_yolox_mnv4_detector_fixture.py")), run_name="__main__")
