#!/usr/bin/env python3
"""Embed a .tflite file as alignas(16) C arrays for MCU firmware."""

from __future__ import annotations

from pathlib import Path


def write_tflite_model_arrays(
    tflite_path: Path,
    *,
    out_h: Path,
    out_cc: Path,
    array_name: str,
) -> None:
    data = tflite_path.read_bytes()
    hex_bytes = ",".join(f"0x{b:02x}" for b in data)

    out_h.parent.mkdir(parents=True, exist_ok=True)
    out_h.write_text(
        "\n".join(
            [
                "#pragma once",
                "",
                "#include <cstdint>",
                "",
                f"constexpr unsigned int {array_name}_size = {len(data)};",
                f"extern const unsigned char {array_name}[];",
                "",
            ]
        ),
        encoding="utf-8",
    )

    include_path = out_h.name
    out_cc.write_text(
        "\n".join(
            [
                "#include <cstdint>",
                "",
                f'#include "{include_path}"',
                "",
                f"alignas(16) const unsigned char {array_name}[] = {{{hex_bytes}}};",
                "",
            ]
        ),
        encoding="utf-8",
    )
