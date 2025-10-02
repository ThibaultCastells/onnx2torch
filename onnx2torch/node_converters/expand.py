__all__ = [
    "OnnxExpand",
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


class OnnxExpand(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-docstring
    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        def _forward() -> torch.Tensor:
            target_shape = shape_tensor_to_sequence(shape)

            if not target_shape:
                if input_tensor.dim() != 0:
                    raise ValueError(
                        "Expand shape must have at least one dimension for non-scalar inputs"  # pragma: no cover - defensive path
                    )
                return input_tensor

            input_rank = input_tensor.dim()
            target_rank = len(target_shape)

            if target_rank < input_rank:
                raise ValueError(
                    "Expand shape has lower rank than the input tensor"  # pragma: no cover - defensive path
                )

            if target_rank > input_rank:
                current_shape = [
                    torch.ops.aten.sym_size.int(input_tensor, index)
                    for index in range(input_rank)
                ]
                reshape_dims = [1] * (target_rank - input_rank) + current_shape
                input_tensor_for_expand = torch.ops.aten.view.default(
                    input_tensor, reshape_dims
                )
            else:
                input_tensor_for_expand = input_tensor

            return input_tensor_for_expand.expand(*target_shape)

        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(
                _forward, "Expand", input_tensor, shape, {}
            )

        return _forward()


@add_converter(operation_type="Expand", version=8)
@add_converter(operation_type="Expand", version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxExpand(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
