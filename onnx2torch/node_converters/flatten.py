import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxFlatten(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, axis: int = 1):
        super().__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        rank = x.dim()
        axis = self.axis
        if axis < 0:
            axis += rank

        axis = max(0, min(axis, rank))

        if axis == 0:
            return x.reshape(1, -1)

        if axis == rank:
            return x.reshape(-1, 1)

        prefix_flat = x.flatten(end_dim=axis - 1)
        return prefix_flat.flatten(start_dim=1)

    @classmethod
    def maybe_create_simple_flatten(cls, axis: int = 1) -> nn.Module:  # pylint: disable=missing-docstring
        return cls(axis=axis)


@add_converter(operation_type="Flatten", version=13)
@add_converter(operation_type="Flatten", version=11)
@add_converter(operation_type="Flatten", version=9)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis = node.attributes.get("axis", 1)
    torch_module = OnnxFlatten.maybe_create_simple_flatten(axis=axis)

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
