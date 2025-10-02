"""Register ONNX converters and expose registry helpers."""

from __future__ import annotations

import importlib
from typing import Sequence

from onnx2torch.node_converters.registry import (
    OperationDescription,
    TConverter,
    get_converter,
)

_CONVERTER_MODULES: Sequence[str] = (
    "onnx2torch.node_converters.activations",
    "onnx2torch.node_converters.arg_extrema",
    "onnx2torch.node_converters.average_pool",
    "onnx2torch.node_converters.batch_norm",
    "onnx2torch.node_converters.binary_math_operations",
    "onnx2torch.node_converters.cast",
    "onnx2torch.node_converters.clip",
    "onnx2torch.node_converters.comparisons",
    "onnx2torch.node_converters.concat",
    "onnx2torch.node_converters.constant",
    "onnx2torch.node_converters.constant_of_shape",
    "onnx2torch.node_converters.conv",
    "onnx2torch.node_converters.cumsum",
    "onnx2torch.node_converters.depth_to_space",
    "onnx2torch.node_converters.dropout",
    "onnx2torch.node_converters.einsum",
    "onnx2torch.node_converters.expand",
    "onnx2torch.node_converters.eye_like",
    "onnx2torch.node_converters.flatten",
    "onnx2torch.node_converters.functions",
    "onnx2torch.node_converters.gather",
    "onnx2torch.node_converters.gemm",
    "onnx2torch.node_converters.global_average_pool",
    "onnx2torch.node_converters.identity",
    "onnx2torch.node_converters.instance_norm",
    "onnx2torch.node_converters.isinf",
    "onnx2torch.node_converters.isnan",
    "onnx2torch.node_converters.layer_norm",
    "onnx2torch.node_converters.logical",
    "onnx2torch.node_converters.lrn",
    "onnx2torch.node_converters.matmul",
    "onnx2torch.node_converters.max_pool",
    "onnx2torch.node_converters.mean",
    "onnx2torch.node_converters.min_max",
    "onnx2torch.node_converters.mod",
    "onnx2torch.node_converters.neg",
    "onnx2torch.node_converters.nms",
    "onnx2torch.node_converters.nonzero",
    "onnx2torch.node_converters.pad",
    "onnx2torch.node_converters.pow",
    "onnx2torch.node_converters.range",
    "onnx2torch.node_converters.reciprocal",
    "onnx2torch.node_converters.reduce",
    "onnx2torch.node_converters.reshape",
    "onnx2torch.node_converters.resize",
    "onnx2torch.node_converters.roialign",
    "onnx2torch.node_converters.roundings",
    "onnx2torch.node_converters.scatter_nd",
    "onnx2torch.node_converters.shape",
    "onnx2torch.node_converters.slice",
    "onnx2torch.node_converters.split",
    "onnx2torch.node_converters.squeeze",
    "onnx2torch.node_converters.sum",
    "onnx2torch.node_converters.tile",
    "onnx2torch.node_converters.topk",
    "onnx2torch.node_converters.transpose",
    "onnx2torch.node_converters.unsqueeze",
    "onnx2torch.node_converters.where",
)

for module_name in _CONVERTER_MODULES:
    importlib.import_module(module_name)

__all__ = ["OperationDescription", "TConverter", "get_converter"]
