# pylint: disable=missing-docstring
__all__ = [
    "OnnxTile",
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport
from onnx2torch.utils.shape_utils import shape_tensor_to_sequence


class OnnxTile(nn.Module, OnnxToTorchModuleWithCustomExport):
    def forward(
        self, input_tensor: torch.Tensor, repeats: torch.Tensor
    ) -> torch.Tensor:
        def _forward() -> torch.Tensor:
            repeat_factors = shape_tensor_to_sequence(repeats)
            if not repeat_factors:
                return input_tensor

            return input_tensor.repeat(*repeat_factors)

        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(
                _forward, "Tile", input_tensor, repeats, {}
            )

        return _forward()


@add_converter(operation_type="Tile", version=6)
@add_converter(operation_type="Tile", version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    del graph
    return OperationConverterResult(
        torch_module=OnnxTile(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
