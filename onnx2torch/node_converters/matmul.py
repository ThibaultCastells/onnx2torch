__all__ = [
    "OnnxMatMul",
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxMatMul(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        if x.dim() == 1 and y.dim() == 1:
            torch._check(x.shape[0] == y.shape[0])
        elif x.dim() == 1:
            torch._check(x.shape[0] == y.shape[-2])
        elif y.dim() == 1:
            torch._check(x.shape[-1] == y.shape[0])
        else:
            torch._check(x.shape[-1] == y.shape[-2])
            x_batch = x.shape[:-2]
            y_batch = y.shape[:-2]
            max_rank = max(len(x_batch), len(y_batch))
            padded_x = (1,) * (max_rank - len(x_batch)) + x_batch
            padded_y = (1,) * (max_rank - len(y_batch)) + y_batch
            for dim_x, dim_y in zip(padded_x, padded_y):
                torch._check((dim_x == dim_y) | (dim_x == 1) | (dim_y == 1))
        return torch.matmul(x, y)


@add_converter(operation_type="MatMul", version=1)
@add_converter(operation_type="MatMul", version=9)
@add_converter(operation_type="MatMul", version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxMatMul(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
