__all__ = ["OnnxExpand"]

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_graph import ValueType
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport
from onnx2torch.utils.shape_utils import shape_tensor_to_sequence


class OnnxExpand(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-docstring
    def __init__(self, static_shape: Optional[Tuple[int, ...]] = None) -> None:
        super().__init__()
        self._static_shape = tuple(static_shape) if static_shape is not None else None

    @staticmethod
    def _sym_size_list(x: torch.Tensor) -> List[Union[int, torch.SymInt]]:
        return [torch.ops.aten.sym_size.int(x, dim) for dim in range(x.dim())]

    @staticmethod
    def _resolve_target_shape_onnx_broadcast(
        input_sizes: List[Union[int, torch.SymInt]],
        raw_shape: List[int],
    ) -> List[Union[int, torch.SymInt]]:
        in_rank = len(input_sizes)
        target_shape = list(raw_shape)

        if len(target_shape) < in_rank:
            target_shape = [1] * (in_rank - len(target_shape)) + target_shape

        if len(target_shape) > in_rank:
            input_sizes = [1] * (len(target_shape) - in_rank) + input_sizes

        resolved: List[Union[int, torch.SymInt]] = []
        for input_dim, target_dim in zip(input_sizes, target_shape):
            if target_dim == 1:
                resolved.append(input_dim)
                continue

            if isinstance(input_dim, int):
                input_is_one = input_dim == 1
                dims_match = input_dim == target_dim
            else:
                input_is_one = input_dim == 1
                dims_match = input_dim == target_dim

            if input_is_one:
                resolved.append(target_dim)
            elif dims_match:
                resolved.append(input_dim)
            else:
                raise ValueError(
                    f"ONNX Expand: incompatible dimensions (input={input_dim}, shape={target_dim})."
                )

        return resolved

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        def _forward() -> torch.Tensor:
            if self._static_shape is not None:
                raw_shape = list(self._static_shape)
            else:
                raw_shape = list(shape_tensor_to_sequence(shape))

            tensor_for_expand = input_tensor
            input_sizes = self._sym_size_list(tensor_for_expand)

            if not raw_shape:
                return input_tensor

            if len(raw_shape) > tensor_for_expand.dim():
                aligned_sizes = [1] * (
                    len(raw_shape) - tensor_for_expand.dim()
                ) + input_sizes
                tensor_for_expand = torch.ops.aten.view.default(
                    tensor_for_expand, aligned_sizes
                )
                input_sizes = aligned_sizes

            final_shape = self._resolve_target_shape_onnx_broadcast(
                input_sizes, raw_shape
            )

            try:
                return tensor_for_expand.expand(*final_shape)
            except RuntimeError:
                zeros = torch.zeros(
                    tuple(final_shape),
                    dtype=tensor_for_expand.dtype,
                    device=tensor_for_expand.device,
                )
                return zeros + tensor_for_expand

        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(
                _forward, "Expand", input_tensor, shape, {}
            )

        return _forward()


@add_converter(operation_type="Expand", version=8)
@add_converter(operation_type="Expand", version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    static_shape: Optional[Tuple[int, ...]] = None
    shape_input_name = node.input_values[1]
    value_type = graph.value_type(shape_input_name)

    if value_type == ValueType.GRAPH_INITIALIZER:
        static_shape = tuple(
            int(dim)
            for dim in graph.initializers[shape_input_name]
            .to_numpy()
            .reshape(-1)
            .tolist()
        )
    elif value_type == ValueType.NODE_OUTPUT:
        producer_node, _ = graph.value_as_node_output(shape_input_name)
        if producer_node.operation_type == "Constant":
            constant_value = producer_node.attributes.get("value")
            if constant_value is not None:
                static_shape = tuple(
                    int(dim) for dim in constant_value.to_numpy().reshape(-1).tolist()
                )

    return OperationConverterResult(
        torch_module=OnnxExpand(static_shape=static_shape),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
