"""CLI: python -m netkit convert|inspect ..."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .onnx_convert import convert_onnx_to_nk
from .inspect import inspect_nk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="netkit model tools (.nk format)")
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="Convert ONNX to .nk")
    convert.add_argument("input", help="Path to .onnx model")
    convert.add_argument("-o", "--output", help="Output .nk path")

    inspect = sub.add_parser("inspect", help="Print .nk header and tensor catalog")
    inspect.add_argument("input", help="Path to .nk model")

    args = parser.parse_args(argv)
    input_path = Path(args.input)

    if args.command == "convert":
        if input_path.suffix.lower() != ".onnx":
            print(f"Unsupported input type: {input_path.suffix} (expected .onnx)", file=sys.stderr)
            return 1
        out = convert_onnx_to_nk(input_path, args.output)
        print(f"Wrote {out}")
        return 0

    if args.command == "inspect":
        inspect_nk(input_path)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
