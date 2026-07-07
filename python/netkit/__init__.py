"""netkit Python tools — convert ONNX models to .nk and inspect them."""

from .aot_compile import AotCompileResult, AotLanguage, compile_aot
from .arch_writer import write_nk_from_arch
from .nk_fuse import FuseArchResult, FuseOptions, fuse_composite_blocks
from .nk_optimize import OptimizeOptions, OptimizeResult, optimize_nk
from .quantize import (
    QuantizedCnnPack,
    QuantizedMlpPack,
    forward_quantized_cnn,
    forward_quantized_mlp,
    quantize_cnn,
    quantize_mlp,
    quantized_cnn_to_spec,
    quantized_mlp_to_spec,
)
from .inspect import inspect_nk
from .reader import read_nk, read_test_suite
from .writer import RegressionCase, RegressionSuite, write_nk

__all__ = [
    "AotCompileResult",
    "AotLanguage",
    "compile_aot",
    "convert_onnx_to_nk",
    "QuantizedCnnPack",
    "QuantizedMlpPack",
    "forward_quantized_cnn",
    "forward_quantized_mlp",
    "quantize_cnn",
    "quantize_mlp",
    "quantized_cnn_to_spec",
    "quantized_mlp_to_spec",
    "fuse_composite_blocks",
    "FuseArchResult",
    "FuseOptions",
    "inspect_nk",
    "optimize_nk",
    "OptimizeOptions",
    "OptimizeResult",
    "read_nk",
    "read_test_suite",
    "write_nk",
    "write_nk_from_arch",
    "RegressionCase",
    "RegressionSuite",
]
